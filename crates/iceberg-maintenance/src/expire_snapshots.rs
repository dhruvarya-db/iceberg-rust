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

use std::collections::HashSet;

use futures::{StreamExt, TryStreamExt, future, stream};
use iceberg::io::FileIO;
use iceberg::spec::{ManifestFile, SnapshotRef, TableMetadata};
use iceberg::table::Table;
use iceberg::transaction::{ApplyTransactionAction, Transaction};
use iceberg::{Catalog, Error, Result};

/// Number of manifests to read concurrently, matching `catalog::utils::drop_table_data`.
const READ_CONCURRENCY: usize = 10;

/// Default upper bound on concurrent file deletions.
const DEFAULT_MAX_CONCURRENT_DELETES: usize = 4;

/// Files that become unreferenced once a set of snapshots is expired.
///
/// A file is unreferenced only if it is reachable from one of the expired snapshots and from no
/// retained snapshot, so files shared with snapshots that survive the expiry are never listed.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct UnreferencedFiles {
    /// Manifest list files (one per expired snapshot).
    pub manifest_lists: Vec<String>,
    /// Manifest files no longer reachable from any retained snapshot.
    pub manifests: Vec<String>,
    /// Data and delete files no longer reachable from any retained snapshot.
    pub data_files: Vec<String>,
    /// Statistics and partition-statistics files of the expired snapshots.
    pub statistics_files: Vec<String>,
}

/// Computes the files that would become unreferenced if `expired_snapshot_ids` were removed.
///
/// This must run against `table`'s metadata *before* the snapshots are removed, while the expired
/// snapshots' manifest lists are still reachable. Data and delete files are reported regardless of
/// the `gc.enabled` property; callers that delete files are responsible for honoring it.
pub async fn unreferenced_files(
    table: &Table,
    expired_snapshot_ids: &HashSet<i64>,
) -> Result<UnreferencedFiles> {
    let io = table.file_io();
    let metadata = table.metadata();

    let retained = reachable_files(io, metadata, |id| !expired_snapshot_ids.contains(&id)).await?;
    let expired = reachable_files(io, metadata, |id| expired_snapshot_ids.contains(&id)).await?;

    let expired_statistics = statistics_paths(metadata, |id| expired_snapshot_ids.contains(&id));
    let retained_statistics = statistics_paths(metadata, |id| !expired_snapshot_ids.contains(&id));

    Ok(UnreferencedFiles {
        manifest_lists: difference(expired.manifest_lists, &retained.manifest_lists),
        manifests: difference(expired.manifests, &retained.manifests),
        data_files: difference(expired.data_files, &retained.data_files),
        statistics_files: difference(expired_statistics, &retained_statistics),
    })
}

/// Statistics and partition-statistics file paths of the snapshots selected by `include`.
fn statistics_paths(metadata: &TableMetadata, include: impl Fn(i64) -> bool) -> HashSet<String> {
    metadata
        .statistics_iter()
        .filter(|file| include(file.snapshot_id))
        .map(|file| file.statistics_path.clone())
        .chain(
            metadata
                .partition_statistics_iter()
                .filter(|file| include(file.snapshot_id))
                .map(|file| file.statistics_path.clone()),
        )
        .collect()
}

/// All files reachable from the snapshots selected by `include`.
struct ReachableFiles {
    manifest_lists: HashSet<String>,
    manifests: HashSet<String>,
    data_files: HashSet<String>,
}

async fn reachable_files(
    io: &FileIO,
    metadata: &TableMetadata,
    include: impl Fn(i64) -> bool,
) -> Result<ReachableFiles> {
    let snapshots: Vec<&SnapshotRef> = metadata
        .snapshots()
        .filter(|snapshot| include(snapshot.snapshot_id()))
        .collect();

    // Load every selected manifest list concurrently, as `drop_table_data` does.
    let manifest_lists = future::try_join_all(snapshots.into_iter().map(|snapshot| async move {
        let manifest_list = snapshot.load_manifest_list(io, metadata).await?;
        Ok::<_, Error>((snapshot.manifest_list().to_string(), manifest_list))
    }))
    .await?;

    let mut manifest_list_paths = HashSet::new();
    let mut manifest_files: Vec<ManifestFile> = Vec::new();
    let mut manifest_paths = HashSet::new();
    for (manifest_list_path, manifest_list) in manifest_lists {
        if !manifest_list_path.is_empty() {
            manifest_list_paths.insert(manifest_list_path);
        }
        for manifest_file in manifest_list.entries() {
            // A manifest can be shared across snapshots; only read it once.
            if manifest_paths.insert(manifest_file.manifest_path.clone()) {
                manifest_files.push(manifest_file.clone());
            }
        }
    }

    let data_files = stream::iter(manifest_files)
        .map(|manifest_file| async move {
            let manifest = manifest_file.load_manifest(io).await?;
            Ok::<Vec<String>, Error>(
                manifest
                    .entries()
                    .iter()
                    .filter(|entry| entry.is_alive())
                    .map(|entry| entry.data_file().file_path().to_string())
                    .collect(),
            )
        })
        .buffer_unordered(READ_CONCURRENCY)
        .try_concat()
        .await?
        .into_iter()
        .collect();

    Ok(ReachableFiles {
        manifest_lists: manifest_list_paths,
        manifests: manifest_paths,
        data_files,
    })
}

fn difference(candidates: HashSet<String>, retained: &HashSet<String>) -> Vec<String> {
    candidates
        .into_iter()
        .filter(|path| !retained.contains(path))
        .collect()
}

/// Outcome of an [`ExpireSnapshots`] run. Counts reflect what was deleted, or what *would* be
/// deleted for a [`dry_run`](ExpireSnapshots::dry_run).
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct ExpireSnapshotsResult {
    /// Snapshots removed from table metadata.
    pub expired_snapshot_ids: Vec<i64>,
    pub deleted_manifest_lists: usize,
    pub deleted_manifests: usize,
    pub deleted_data_files: usize,
    pub deleted_statistics_files: usize,
    /// Whether this was a preview that performed no commit or deletion.
    pub dry_run: bool,
}

/// Expires table snapshots and deletes the files they leave behind.
///
/// This both commits a new metadata version with the snapshots removed (via the core
/// [`expire_snapshots`](iceberg::transaction::Transaction::expire_snapshots) action) and deletes
/// the now-unreferenced files. Selection follows the same rules as that action: explicit ids, or
/// age plus `retain_last`, never the current snapshot.
///
/// Data and delete files are only removed when the `gc.enabled` table property is set, mirroring
/// [`iceberg::drop_table_data`].
pub struct ExpireSnapshots {
    table: Table,
    snapshot_ids: Vec<i64>,
    older_than_ms: Option<i64>,
    retain_last: Option<usize>,
    dry_run: bool,
    max_concurrent_deletes: usize,
}

impl ExpireSnapshots {
    pub fn new(table: Table) -> Self {
        Self {
            table,
            snapshot_ids: vec![],
            older_than_ms: None,
            retain_last: None,
            dry_run: false,
            max_concurrent_deletes: DEFAULT_MAX_CONCURRENT_DELETES,
        }
    }

    /// Expire exactly these snapshot ids, ignoring the age and `retain_last` filters.
    pub fn expire_snapshot_ids(mut self, snapshot_ids: Vec<i64>) -> Self {
        self.snapshot_ids = snapshot_ids;
        self
    }

    /// Expire snapshots whose timestamp is strictly older than `older_than_ms`.
    pub fn expire_older_than_ms(mut self, older_than_ms: i64) -> Self {
        self.older_than_ms = Some(older_than_ms);
        self
    }

    /// Retain at least the `retain_last` most recent snapshots (defaults to 1).
    pub fn retain_last(mut self, retain_last: usize) -> Self {
        self.retain_last = Some(retain_last);
        self
    }

    /// Preview the operation without committing or deleting anything.
    pub fn dry_run(mut self, dry_run: bool) -> Self {
        self.dry_run = dry_run;
        self
    }

    /// Upper bound on concurrent file deletions (clamped to at least 1).
    pub fn max_concurrent_deletes(mut self, max_concurrent_deletes: usize) -> Self {
        self.max_concurrent_deletes = max_concurrent_deletes.max(1);
        self
    }

    /// Snapshot ids that would be expired, reusing the core action's selection rules.
    fn snapshot_ids_to_expire(&self) -> Result<Vec<i64>> {
        // The action type is reachable only through `Transaction`, so build it inline.
        let mut action = Transaction::new(&self.table).expire_snapshots();
        if !self.snapshot_ids.is_empty() {
            action = action.expire_snapshot_ids(self.snapshot_ids.clone());
        }
        if let Some(older_than_ms) = self.older_than_ms {
            action = action.expire_older_than_ms(older_than_ms);
        }
        if let Some(retain_last) = self.retain_last {
            action = action.retain_last(retain_last);
        }
        action.snapshot_ids_to_expire(&self.table)
    }

    pub async fn execute(self, catalog: &dyn Catalog) -> Result<ExpireSnapshotsResult> {
        let expired_ids = self.snapshot_ids_to_expire()?;
        if expired_ids.is_empty() {
            return Ok(ExpireSnapshotsResult {
                dry_run: self.dry_run,
                ..Default::default()
            });
        }

        let expired_set: HashSet<i64> = expired_ids.iter().copied().collect();
        let files = unreferenced_files(&self.table, &expired_set).await?;
        let gc_enabled = self.table.metadata().table_properties()?.gc_enabled;

        let result = ExpireSnapshotsResult {
            expired_snapshot_ids: expired_ids.clone(),
            deleted_manifest_lists: files.manifest_lists.len(),
            deleted_manifests: files.manifests.len(),
            deleted_data_files: if gc_enabled {
                files.data_files.len()
            } else {
                0
            },
            deleted_statistics_files: files.statistics_files.len(),
            dry_run: self.dry_run,
        };

        if self.dry_run {
            return Ok(result);
        }

        // Commit the removal with explicit ids so the committed set matches the files computed
        // above. File deletion only happens after the metadata commit succeeds.
        let tx = Transaction::new(&self.table);
        let action = tx.expire_snapshots().expire_snapshot_ids(expired_ids);
        action.apply(tx)?.commit(catalog).await?;

        let mut to_delete = files.manifest_lists;
        to_delete.extend(files.manifests);
        // Statistics files belong to a single snapshot and are never shared across tables, so
        // they are deleted regardless of `gc.enabled` (unlike data files).
        to_delete.extend(files.statistics_files);
        if gc_enabled {
            to_delete.extend(files.data_files);
        }
        self.delete_files(to_delete).await?;

        Ok(result)
    }

    async fn delete_files(&self, paths: Vec<String>) -> Result<()> {
        let io = self.table.file_io().clone();
        stream::iter(paths.into_iter().map(Ok::<_, Error>))
            .try_for_each_concurrent(self.max_concurrent_deletes, |path| {
                let io = io.clone();
                async move { io.delete(&path).await }
            })
            .await
    }
}

#[cfg(test)]
mod tests {
    use std::collections::{HashMap, HashSet};

    use iceberg::memory::{MEMORY_CATALOG_WAREHOUSE, MemoryCatalogBuilder};
    use iceberg::spec::{
        DataContentType, DataFileBuilder, DataFileFormat, FormatVersion, StatisticsFile, Struct,
    };
    use iceberg::transaction::{ApplyTransactionAction, Transaction};
    use iceberg::{Catalog, CatalogBuilder, NamespaceIdent, TableCreation};

    use super::*;

    async fn memory_catalog() -> impl Catalog {
        let warehouse = tempfile::tempdir()
            .unwrap()
            .keep()
            .to_str()
            .unwrap()
            .to_string();
        MemoryCatalogBuilder::default()
            .load(
                "memory",
                HashMap::from([(MEMORY_CATALOG_WAREHOUSE.to_string(), warehouse)]),
            )
            .await
            .unwrap()
    }

    fn data_file(path: &str) -> iceberg::spec::DataFile {
        DataFileBuilder::default()
            .content(DataContentType::Data)
            .file_path(path.to_string())
            .file_format(DataFileFormat::Parquet)
            .file_size_in_bytes(100)
            .record_count(1)
            .partition(Struct::empty())
            .partition_spec_id(0)
            .build()
            .unwrap()
    }

    async fn empty_table(catalog: &impl Catalog) -> Table {
        let namespace = NamespaceIdent::new(format!("ns-{}", uuid::Uuid::now_v7()));
        catalog
            .create_namespace(&namespace, HashMap::new())
            .await
            .unwrap();
        let creation = TableCreation::builder()
            .name("t".to_string())
            .schema(iceberg::spec::Schema::builder().build().unwrap())
            .format_version(FormatVersion::V2)
            .build();
        catalog.create_table(&namespace, creation).await.unwrap()
    }

    async fn append(catalog: &impl Catalog, table: &Table, path: &str) -> Table {
        let tx = Transaction::new(table);
        tx.fast_append()
            .add_data_files(vec![data_file(path)])
            .apply(tx)
            .unwrap()
            .commit(catalog)
            .await
            .unwrap()
    }

    fn current_id(table: &Table) -> i64 {
        table.metadata().current_snapshot_id().unwrap()
    }

    async fn set_statistics(
        catalog: &impl Catalog,
        table: &Table,
        snapshot_id: i64,
        path: &str,
    ) -> Table {
        let tx = Transaction::new(table);
        tx.update_statistics()
            .set_statistics(StatisticsFile {
                snapshot_id,
                statistics_path: path.to_string(),
                file_size_in_bytes: 1,
                file_footer_size_in_bytes: 1,
                key_metadata: None,
                blob_metadata: vec![],
            })
            .apply(tx)
            .unwrap()
            .commit(catalog)
            .await
            .unwrap()
    }

    #[tokio::test]
    async fn test_expiring_older_snapshot_keeps_carried_forward_files() {
        let catalog = memory_catalog().await;
        let table = empty_table(&catalog).await;
        let table = append(&catalog, &table, "data/1.parquet").await;
        let first = current_id(&table);
        let table = append(&catalog, &table, "data/2.parquet").await;

        let files = unreferenced_files(&table, &HashSet::from([first]))
            .await
            .unwrap();

        // Fast append carries the older snapshot's manifest and data file forward into the
        // retained snapshot, so only the older snapshot's own manifest list is freed.
        assert_eq!(files.manifest_lists.len(), 1);
        assert!(files.manifests.is_empty());
        assert!(files.data_files.is_empty());
    }

    #[tokio::test]
    async fn test_expiring_newer_snapshot_frees_its_files() {
        let catalog = memory_catalog().await;
        let table = empty_table(&catalog).await;
        let table = append(&catalog, &table, "data/1.parquet").await;
        let table = append(&catalog, &table, "data/2.parquet").await;
        let second = current_id(&table);

        // Expiring the newer snapshot must not delete the older snapshot's still-live files.
        let files = unreferenced_files(&table, &HashSet::from([second]))
            .await
            .unwrap();

        assert_eq!(files.data_files, vec!["data/2.parquet".to_string()]);
        assert!(!files.manifests.is_empty());
        assert_eq!(files.manifest_lists.len(), 1);
    }

    #[tokio::test]
    async fn test_no_expired_snapshots_is_empty() {
        let catalog = memory_catalog().await;
        let table = empty_table(&catalog).await;
        let table = append(&catalog, &table, "data/1.parquet").await;

        let files = unreferenced_files(&table, &HashSet::new()).await.unwrap();
        assert_eq!(files, UnreferencedFiles::default());
    }

    fn manifest_list_path(table: &Table, snapshot_id: i64) -> String {
        table
            .metadata()
            .snapshot_by_id(snapshot_id)
            .unwrap()
            .manifest_list()
            .to_string()
    }

    #[tokio::test]
    async fn test_execute_expires_snapshots_and_deletes_files() {
        let catalog = memory_catalog().await;
        let table = empty_table(&catalog).await;
        let table = append(&catalog, &table, "data/1.parquet").await;
        let oldest = current_id(&table);
        let table = append(&catalog, &table, "data/2.parquet").await;
        let table = append(&catalog, &table, "data/3.parquet").await;
        let current = current_id(&table);

        let io = table.file_io().clone();
        let oldest_manifest_list = manifest_list_path(&table, oldest);
        assert!(io.exists(&oldest_manifest_list).await.unwrap());

        let result = ExpireSnapshots::new(table.clone())
            .retain_last(1)
            .expire_older_than_ms(i64::MAX)
            .execute(&catalog)
            .await
            .unwrap();

        assert_eq!(result.expired_snapshot_ids.len(), 2);
        assert!(result.deleted_manifest_lists >= 1);
        assert!(!result.dry_run);
        assert!(!io.exists(&oldest_manifest_list).await.unwrap());

        let reloaded = catalog.load_table(table.identifier()).await.unwrap();
        assert_eq!(reloaded.metadata().snapshots().count(), 1);
        assert_eq!(reloaded.metadata().current_snapshot_id(), Some(current));
    }

    #[tokio::test]
    async fn test_dry_run_commits_and_deletes_nothing() {
        let catalog = memory_catalog().await;
        let table = empty_table(&catalog).await;
        let table = append(&catalog, &table, "data/1.parquet").await;
        let oldest = current_id(&table);
        let table = append(&catalog, &table, "data/2.parquet").await;

        let io = table.file_io().clone();
        let oldest_manifest_list = manifest_list_path(&table, oldest);

        let result = ExpireSnapshots::new(table.clone())
            .retain_last(1)
            .expire_older_than_ms(i64::MAX)
            .dry_run(true)
            .execute(&catalog)
            .await
            .unwrap();

        assert!(result.dry_run);
        assert_eq!(result.expired_snapshot_ids, vec![oldest]);
        assert!(result.deleted_manifest_lists >= 1);

        assert!(io.exists(&oldest_manifest_list).await.unwrap());
        let reloaded = catalog.load_table(table.identifier()).await.unwrap();
        assert_eq!(reloaded.metadata().snapshots().count(), 2);
    }

    #[tokio::test]
    async fn test_expire_removes_statistics_of_expired_snapshot() {
        let catalog = memory_catalog().await;
        let table = empty_table(&catalog).await;
        let table = append(&catalog, &table, "data/1.parquet").await;
        let oldest = current_id(&table);
        let table = set_statistics(&catalog, &table, oldest, "stats/oldest.puffin").await;
        let table = append(&catalog, &table, "data/2.parquet").await;

        let files = unreferenced_files(&table, &HashSet::from([oldest]))
            .await
            .unwrap();
        assert_eq!(files.statistics_files, vec![
            "stats/oldest.puffin".to_string()
        ]);

        let result = ExpireSnapshots::new(table.clone())
            .expire_snapshot_ids(vec![oldest])
            .execute(&catalog)
            .await
            .unwrap();

        assert_eq!(result.deleted_statistics_files, 1);
        let reloaded = catalog.load_table(table.identifier()).await.unwrap();
        assert!(
            reloaded
                .metadata()
                .statistics_for_snapshot(oldest)
                .is_none()
        );
    }
}
