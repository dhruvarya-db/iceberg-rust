// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

//! Pure-Rust, engine-agnostic table-maintenance building blocks for Apache Iceberg.
//!
//! This crate intentionally only computes *what* to delete; committing the metadata change and
//! performing the deletion is left to a higher-level orchestrator (see issue
//! [#2145](https://github.com/apache/iceberg-rust/issues/2145)).

use std::collections::HashSet;

use futures::StreamExt;
use iceberg::Result;
use iceberg::spec::{DataContentType, Manifest, SnapshotRef, TableMetadata};
use iceberg::table::Table;

/// Bound on concurrent manifest-list / manifest loads, matching `CatalogUtil`'s delete concurrency.
const LOAD_CONCURRENCY: usize = 10;

/// Files reachable only from expired snapshots, grouped by kind, that are therefore safe to delete.
///
/// Paths are absolute and de-duplicated. A file referenced by any retained snapshot is never
/// included, even if an expired snapshot also references it.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct UnreferencedFiles {
    /// Manifest-list (snapshot) files.
    pub manifest_lists: HashSet<String>,
    /// Manifest files.
    pub manifests: HashSet<String>,
    /// Data files (`DataContentType::Data`).
    pub data_files: HashSet<String>,
    /// Delete files (positional or equality deletes).
    pub delete_files: HashSet<String>,
    /// Table statistics (Puffin) files.
    pub statistics_files: HashSet<String>,
    /// Partition statistics files.
    pub partition_statistics_files: HashSet<String>,
}

impl UnreferencedFiles {
    /// Total number of files across every kind.
    pub fn len(&self) -> usize {
        self.manifest_lists.len()
            + self.manifests.len()
            + self.data_files.len()
            + self.delete_files.len()
            + self.statistics_files.len()
            + self.partition_statistics_files.len()
    }

    /// Whether there is nothing to delete.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Iterates every file path across all kinds.
    pub fn all_paths(&self) -> impl Iterator<Item = &String> {
        self.manifest_lists
            .iter()
            .chain(&self.manifests)
            .chain(&self.data_files)
            .chain(&self.delete_files)
            .chain(&self.statistics_files)
            .chain(&self.partition_statistics_files)
    }
}

/// Computes the files reachable only from the expired snapshots of `table`.
///
/// `table` must be the metadata *before* expiry (still carrying the expired snapshots), and
/// `expired_snapshot_ids` the ids about to be removed. The result is the reference-count difference
/// `files(expired) − files(retained)`, mirroring Java `ReachableFileCleanup`: a file kept alive by
/// any surviving snapshot is excluded.
///
/// Data and delete files are only collected when the `gc.enabled` table property is `true`, matching
/// [`iceberg::catalog::utils::drop_table_data`] — they may be shared with other tables (e.g. via
/// shallow clones), whereas manifests, manifest lists, and statistics files are table-private.
///
/// Reachability is resolved by reading manifests, so an I/O error matters: a failure to load a
/// **retained** snapshot's manifest list, or to read one of its manifests, aborts the whole call
/// (deleting a file we could not prove unreferenced would be unsafe). The same failure for an
/// **expired** snapshot is skipped — that snapshot simply contributes nothing, leaving its files for
/// a later orphan-cleanup pass.
///
/// This is metadata analysis only; no files are deleted.
pub async fn unreferenced_files(
    table: &Table,
    expired_snapshot_ids: &HashSet<i64>,
) -> Result<UnreferencedFiles> {
    let metadata = table.metadata();
    let gc_enabled = metadata.table_properties()?.gc_enabled;

    let (expired, retained): (Vec<&SnapshotRef>, Vec<&SnapshotRef>) = metadata
        .snapshots()
        .partition(|snapshot| expired_snapshot_ids.contains(&snapshot.snapshot_id()));

    let retained_reachable = collect_reachable(table, &retained, gc_enabled, OnError::Fail).await?;
    let expired_reachable = collect_reachable(table, &expired, gc_enabled, OnError::Skip).await?;

    let (retained_stats, retained_partition_stats) = stats_paths(metadata, &retained);
    let (expired_stats, expired_partition_stats) = stats_paths(metadata, &expired);

    Ok(UnreferencedFiles {
        manifest_lists: difference(
            expired_reachable.manifest_lists,
            &retained_reachable.manifest_lists,
        ),
        manifests: difference(expired_reachable.manifests, &retained_reachable.manifests),
        data_files: difference(expired_reachable.data_files, &retained_reachable.data_files),
        delete_files: difference(
            expired_reachable.delete_files,
            &retained_reachable.delete_files,
        ),
        statistics_files: difference(expired_stats, &retained_stats),
        partition_statistics_files: difference(expired_partition_stats, &retained_partition_stats),
    })
}

/// How a manifest-list / manifest load failure is handled while collecting a reachable set.
#[derive(Clone, Copy)]
enum OnError {
    /// Abort: used for retained snapshots, whose files must never be mistaken for unreferenced.
    Fail,
    /// Skip the offending snapshot/manifest: used for expired snapshots (best-effort cleanup).
    Skip,
}

/// File paths reachable from a set of snapshots, before any anti-join against the retained set.
#[derive(Default)]
struct Reachable {
    manifest_lists: HashSet<String>,
    manifests: HashSet<String>,
    data_files: HashSet<String>,
    delete_files: HashSet<String>,
}

async fn collect_reachable(
    table: &Table,
    snapshots: &[&SnapshotRef],
    gc_enabled: bool,
    on_error: OnError,
) -> Result<Reachable> {
    let mut reachable = Reachable::default();

    // Manifest lists -> manifest paths.
    let manifest_list_loads = futures::stream::iter(snapshots.iter().copied())
        .map(|snapshot| async move {
            let manifest_list = table.manifest_list_reader(snapshot).load().await?;
            Ok::<_, iceberg::Error>((snapshot.manifest_list().to_string(), manifest_list))
        })
        .buffer_unordered(LOAD_CONCURRENCY)
        .collect::<Vec<_>>()
        .await;

    let mut manifest_paths: HashSet<String> = HashSet::new();
    for load in manifest_list_loads {
        let (location, manifest_list) = match load {
            Ok(loaded) => loaded,
            Err(e) => match on_error {
                OnError::Fail => return Err(e),
                OnError::Skip => continue,
            },
        };
        if !location.is_empty() {
            reachable.manifest_lists.insert(location);
        }
        for manifest_file in manifest_list.entries() {
            reachable
                .manifests
                .insert(manifest_file.manifest_path.clone());
            manifest_paths.insert(manifest_file.manifest_path.clone());
        }
    }

    // Data/delete files are only relevant under gc.enabled (see the function-level docs).
    if gc_enabled {
        let io = table.file_io();
        let manifest_reads = futures::stream::iter(manifest_paths)
            .map(|path| async move {
                let bytes = io.new_input(&path)?.read().await?;
                Manifest::parse_avro(&bytes)
            })
            .buffer_unordered(LOAD_CONCURRENCY)
            .collect::<Vec<_>>()
            .await;

        for read in manifest_reads {
            let manifest = match read {
                Ok(manifest) => manifest,
                Err(e) => match on_error {
                    OnError::Fail => return Err(e),
                    OnError::Skip => continue,
                },
            };
            for entry in manifest.entries() {
                let path = entry.file_path().to_string();
                match entry.data_file().content_type() {
                    DataContentType::Data => reachable.data_files.insert(path),
                    DataContentType::PositionDeletes | DataContentType::EqualityDeletes => {
                        reachable.delete_files.insert(path)
                    }
                };
            }
        }
    }

    Ok(reachable)
}

/// Statistics and partition-statistics file paths for the given snapshots (keyed by snapshot id).
fn stats_paths(
    metadata: &TableMetadata,
    snapshots: &[&SnapshotRef],
) -> (HashSet<String>, HashSet<String>) {
    let mut statistics = HashSet::new();
    let mut partition_statistics = HashSet::new();
    for snapshot in snapshots {
        if let Some(file) = metadata.statistics_for_snapshot(snapshot.snapshot_id()) {
            statistics.insert(file.statistics_path.clone());
        }
        if let Some(file) = metadata.partition_statistics_for_snapshot(snapshot.snapshot_id()) {
            partition_statistics.insert(file.statistics_path.clone());
        }
    }
    (statistics, partition_statistics)
}

/// `from − remove`, consuming `from`.
fn difference(mut from: HashSet<String>, remove: &HashSet<String>) -> HashSet<String> {
    from.retain(|path| !remove.contains(path));
    from
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::sync::Arc;

    use iceberg::TableIdent;
    use iceberg::io::FileIO;
    use iceberg::spec::{
        DataContentType, DataFileBuilder, DataFileFormat, FormatVersion, ManifestFile,
        ManifestListWriter, ManifestWriterBuilder, NestedField, Operation, PartitionSpec,
        PrimitiveType, Schema, SchemaRef, Snapshot, SortOrder, StatisticsFile, Struct, Summary,
        TableMetadataBuilder, Type,
    };
    use iceberg::table::Table;
    use tempfile::TempDir;
    use uuid::Uuid;

    use super::*;

    const OLD: i64 = 1;
    const CURRENT: i64 = 2;

    fn schema() -> SchemaRef {
        Arc::new(
            Schema::builder()
                .with_schema_id(0)
                .with_fields(vec![
                    NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int)).into(),
                ])
                .build()
                .unwrap(),
        )
    }

    fn data_file(path: &str, content: DataContentType) -> iceberg::spec::DataFile {
        DataFileBuilder::default()
            .partition_spec_id(0)
            .content(content)
            .file_path(path.to_string())
            .file_format(DataFileFormat::Parquet)
            .file_size_in_bytes(10)
            .record_count(1)
            .partition(Struct::empty())
            .key_metadata(None)
            .build()
            .unwrap()
    }

    /// Writes a manifest holding `files` (all of `content` kind) and returns it.
    async fn write_manifest(
        file_io: &FileIO,
        table_location: &str,
        snapshot_id: i64,
        sequence_number: i64,
        content: DataContentType,
        files: &[&str],
    ) -> ManifestFile {
        let output = file_io
            .new_output(format!(
                "{table_location}/metadata/manifest-{snapshot_id}-{}.avro",
                Uuid::new_v4()
            ))
            .unwrap();
        let builder =
            ManifestWriterBuilder::new(output, Some(snapshot_id), None, schema(), unpartitioned());
        let mut writer = match content {
            DataContentType::Data => builder.build_v2_data(),
            _ => builder.build_v2_deletes(),
        };
        for path in files {
            writer
                .add_file(data_file(path, content), sequence_number)
                .unwrap();
        }
        writer.write_manifest_file().await.unwrap()
    }

    /// Writes a snapshot's manifest list (referencing optional data/delete manifests) and returns its
    /// location. The manifest-list file is only created here; callers that want a *missing* list pass
    /// the returned location to the snapshot without calling this.
    async fn write_manifest_list(
        file_io: &FileIO,
        table_location: &str,
        snapshot_id: i64,
        parent_snapshot_id: Option<i64>,
        sequence_number: i64,
        data_files: &[&str],
        delete_files: &[&str],
    ) -> String {
        let mut manifests: Vec<ManifestFile> = vec![];
        if !data_files.is_empty() {
            manifests.push(
                write_manifest(
                    file_io,
                    table_location,
                    snapshot_id,
                    sequence_number,
                    DataContentType::Data,
                    data_files,
                )
                .await,
            );
        }
        if !delete_files.is_empty() {
            manifests.push(
                write_manifest(
                    file_io,
                    table_location,
                    snapshot_id,
                    sequence_number,
                    DataContentType::PositionDeletes,
                    delete_files,
                )
                .await,
            );
        }

        let location = format!("{table_location}/metadata/snap-{snapshot_id}.avro");
        let output = file_io.new_output(&location).unwrap();
        let mut writer = ManifestListWriter::v2(
            output.writer().await.unwrap(),
            snapshot_id,
            parent_snapshot_id,
            sequence_number,
        );
        writer.add_manifests(manifests.into_iter()).unwrap();
        writer.close().await.unwrap();
        location
    }

    fn unpartitioned() -> PartitionSpec {
        PartitionSpec::unpartition_spec()
    }

    fn snapshot(
        id: i64,
        parent: Option<i64>,
        sequence_number: i64,
        timestamp_ms: i64,
        manifest_list: String,
    ) -> Snapshot {
        Snapshot::builder()
            .with_snapshot_id(id)
            .with_parent_snapshot_id(parent)
            .with_sequence_number(sequence_number)
            .with_timestamp_ms(timestamp_ms)
            .with_schema_id(0)
            .with_manifest_list(manifest_list)
            .with_summary(Summary {
                operation: Operation::Append,
                additional_properties: HashMap::new(),
            })
            .build()
    }

    /// Builds a table from pre-built (old, current) snapshots, with `current` as the main head.
    fn table_with(
        tmp: &TempDir,
        file_io: FileIO,
        old: Snapshot,
        current: Snapshot,
        properties: HashMap<String, String>,
        statistics: Vec<StatisticsFile>,
    ) -> Table {
        let location = tmp.path().join("table").to_str().unwrap().to_string();
        let mut builder = TableMetadataBuilder::new(
            (*schema()).clone(),
            unpartitioned(),
            SortOrder::unsorted_order(),
            location,
            FormatVersion::V2,
            properties,
        )
        .unwrap()
        .add_snapshot(old)
        .unwrap()
        .add_snapshot(current.clone())
        .unwrap()
        .set_ref("main", iceberg::spec::SnapshotReference {
            snapshot_id: current.snapshot_id(),
            retention: iceberg::spec::SnapshotRetention::Branch {
                min_snapshots_to_keep: None,
                max_snapshot_age_ms: None,
                max_ref_age_ms: None,
            },
        })
        .unwrap();
        for stats in statistics {
            builder = builder.set_statistics(stats);
        }
        let metadata = builder.build().unwrap().metadata;

        Table::builder()
            .metadata(metadata)
            .identifier(TableIdent::from_strs(["db", "t"]).unwrap())
            .file_io(file_io)
            .metadata_location(
                tmp.path()
                    .join("metadata/v1.json")
                    .to_str()
                    .unwrap()
                    .to_string(),
            )
            .runtime(iceberg::test_utils::test_runtime())
            .build()
            .unwrap()
    }

    fn stats_file(snapshot_id: i64, path: &str) -> StatisticsFile {
        StatisticsFile {
            snapshot_id,
            statistics_path: path.to_string(),
            file_size_in_bytes: 1,
            file_footer_size_in_bytes: 1,
            key_metadata: None,
            blob_metadata: vec![],
        }
    }

    #[tokio::test]
    async fn returns_only_files_reachable_solely_from_expired_snapshot() {
        let tmp = TempDir::new().unwrap();
        let file_io = FileIO::new_with_fs();
        let loc = tmp.path().join("table").to_str().unwrap().to_string();

        // OLD shares `shared.parquet` with CURRENT; each also has a private data file.
        let old_list = write_manifest_list(
            &file_io,
            &loc,
            OLD,
            None,
            1,
            &["/shared.parquet", "/old.parquet"],
            &[],
        )
        .await;
        let cur_list = write_manifest_list(
            &file_io,
            &loc,
            CURRENT,
            Some(OLD),
            2,
            &["/shared.parquet", "/cur.parquet"],
            &[],
        )
        .await;

        let table = table_with(
            &tmp,
            file_io,
            snapshot(OLD, None, 1, 1000, old_list.clone()),
            snapshot(CURRENT, Some(OLD), 2, 2000, cur_list),
            HashMap::new(),
            vec![],
        );

        let files = unreferenced_files(&table, &HashSet::from([OLD]))
            .await
            .unwrap();

        assert_eq!(files.manifest_lists, HashSet::from([old_list]));
        assert_eq!(
            files.data_files,
            HashSet::from(["/old.parquet".to_string()])
        );
        assert!(files.delete_files.is_empty());
        // OLD has exactly one (data) manifest, distinct from CURRENT's.
        assert_eq!(files.manifests.len(), 1);
    }

    #[tokio::test]
    async fn returns_delete_files_of_expired_snapshot() {
        let tmp = TempDir::new().unwrap();
        let file_io = FileIO::new_with_fs();
        let loc = tmp.path().join("table").to_str().unwrap().to_string();

        let old_list = write_manifest_list(&file_io, &loc, OLD, None, 1, &["/old.parquet"], &[
            "/old-delete.parquet",
        ])
        .await;
        let cur_list =
            write_manifest_list(&file_io, &loc, CURRENT, Some(OLD), 2, &["/cur.parquet"], &[
            ])
            .await;

        let table = table_with(
            &tmp,
            file_io,
            snapshot(OLD, None, 1, 1000, old_list),
            snapshot(CURRENT, Some(OLD), 2, 2000, cur_list),
            HashMap::new(),
            vec![],
        );

        let files = unreferenced_files(&table, &HashSet::from([OLD]))
            .await
            .unwrap();
        assert_eq!(
            files.data_files,
            HashSet::from(["/old.parquet".to_string()])
        );
        assert_eq!(
            files.delete_files,
            HashSet::from(["/old-delete.parquet".to_string()])
        );
    }

    #[tokio::test]
    async fn returns_statistics_files_of_expired_snapshot() {
        let tmp = TempDir::new().unwrap();
        let file_io = FileIO::new_with_fs();
        let loc = tmp.path().join("table").to_str().unwrap().to_string();

        let old_list =
            write_manifest_list(&file_io, &loc, OLD, None, 1, &["/old.parquet"], &[]).await;
        let cur_list =
            write_manifest_list(&file_io, &loc, CURRENT, Some(OLD), 2, &["/cur.parquet"], &[
            ])
            .await;

        let table = table_with(
            &tmp,
            file_io,
            snapshot(OLD, None, 1, 1000, old_list),
            snapshot(CURRENT, Some(OLD), 2, 2000, cur_list),
            HashMap::new(),
            vec![
                stats_file(OLD, "/old-stats.puffin"),
                stats_file(CURRENT, "/cur-stats.puffin"),
            ],
        );

        let files = unreferenced_files(&table, &HashSet::from([OLD]))
            .await
            .unwrap();
        // Only the expired snapshot's statistics file is returned; the retained one is kept.
        assert_eq!(
            files.statistics_files,
            HashSet::from(["/old-stats.puffin".to_string()])
        );
    }

    #[tokio::test]
    async fn gc_disabled_excludes_content_files_but_keeps_metadata() {
        let tmp = TempDir::new().unwrap();
        let file_io = FileIO::new_with_fs();
        let loc = tmp.path().join("table").to_str().unwrap().to_string();

        let old_list =
            write_manifest_list(&file_io, &loc, OLD, None, 1, &["/old.parquet"], &[]).await;
        let cur_list =
            write_manifest_list(&file_io, &loc, CURRENT, Some(OLD), 2, &["/cur.parquet"], &[
            ])
            .await;

        let table = table_with(
            &tmp,
            file_io,
            snapshot(OLD, None, 1, 1000, old_list.clone()),
            snapshot(CURRENT, Some(OLD), 2, 2000, cur_list),
            HashMap::from([("gc.enabled".to_string(), "false".to_string())]),
            vec![],
        );

        let files = unreferenced_files(&table, &HashSet::from([OLD]))
            .await
            .unwrap();
        // Data files are not collected with gc disabled, but the table-private metadata still is.
        assert!(files.data_files.is_empty());
        assert_eq!(files.manifest_lists, HashSet::from([old_list]));
        assert_eq!(files.manifests.len(), 1);
    }

    #[tokio::test]
    async fn retained_snapshot_load_error_fails() {
        let tmp = TempDir::new().unwrap();
        let file_io = FileIO::new_with_fs();
        let loc = tmp.path().join("table").to_str().unwrap().to_string();

        // CURRENT (retained) points at a manifest list that was never written.
        let old_list =
            write_manifest_list(&file_io, &loc, OLD, None, 1, &["/old.parquet"], &[]).await;
        let missing = format!("{loc}/metadata/snap-missing.avro");

        let table = table_with(
            &tmp,
            file_io,
            snapshot(OLD, None, 1, 1000, old_list),
            snapshot(CURRENT, Some(OLD), 2, 2000, missing),
            HashMap::new(),
            vec![],
        );

        assert!(
            unreferenced_files(&table, &HashSet::from([OLD]))
                .await
                .is_err()
        );
    }

    #[tokio::test]
    async fn expired_snapshot_load_error_is_skipped() {
        let tmp = TempDir::new().unwrap();
        let file_io = FileIO::new_with_fs();
        let loc = tmp.path().join("table").to_str().unwrap().to_string();

        // OLD (expired) points at a manifest list that was never written; CURRENT is intact.
        let missing = format!("{loc}/metadata/snap-missing.avro");
        let cur_list =
            write_manifest_list(&file_io, &loc, CURRENT, Some(OLD), 2, &["/cur.parquet"], &[
            ])
            .await;

        let table = table_with(
            &tmp,
            file_io,
            snapshot(OLD, None, 1, 1000, missing),
            snapshot(CURRENT, Some(OLD), 2, 2000, cur_list),
            HashMap::new(),
            vec![],
        );

        // The unreadable expired snapshot is skipped, so nothing is reported for deletion.
        let files = unreferenced_files(&table, &HashSet::from([OLD]))
            .await
            .unwrap();
        assert!(files.is_empty());
    }
}
