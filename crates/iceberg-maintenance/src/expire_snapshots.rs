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
use iceberg::{Error, Result};

/// Number of manifests to read concurrently, matching `catalog::utils::drop_table_data`.
const READ_CONCURRENCY: usize = 10;

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

    Ok(UnreferencedFiles {
        manifest_lists: difference(expired.manifest_lists, &retained.manifest_lists),
        manifests: difference(expired.manifests, &retained.manifests),
        data_files: difference(expired.data_files, &retained.data_files),
    })
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

#[cfg(test)]
mod tests {
    use std::collections::{HashMap, HashSet};

    use iceberg::memory::{MEMORY_CATALOG_WAREHOUSE, MemoryCatalogBuilder};
    use iceberg::spec::{DataContentType, DataFileBuilder, DataFileFormat, FormatVersion, Struct};
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
}
