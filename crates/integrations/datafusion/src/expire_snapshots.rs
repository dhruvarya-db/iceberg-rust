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

use futures::{StreamExt, TryStreamExt};
use iceberg::io::FileIO;
use iceberg::table::Table;
use iceberg::transaction::{ApplyTransactionAction, Transaction};
use iceberg::{Catalog, Result};
use iceberg_maintenance::{UnreferencedFiles, unreferenced_files};

/// Default bound on concurrent file deletions.
const DEFAULT_MAX_CONCURRENT_DELETES: usize = 4;

/// Expires snapshots and deletes the files they leave unreferenced.
///
/// This is the orchestrator on top of the core [`ExpireSnapshotsAction`] and
/// [`iceberg_maintenance::unreferenced_files`]: it selects the snapshots to expire, commits the
/// metadata change through a [`Transaction`], and then deletes the now-unreferenced files. The
/// selection knobs forward to the core action; see it for their exact semantics.
///
/// [`ExpireSnapshotsAction`]: iceberg::transaction::ExpireSnapshotsAction
pub struct ExpireSnapshots {
    table: Table,
    snapshot_ids: Vec<i64>,
    older_than_ms: Option<i64>,
    retain_last: Option<usize>,
    max_concurrent_deletes: usize,
    dry_run: bool,
}

impl ExpireSnapshots {
    /// Starts an expire-snapshots operation against `table`.
    pub fn new(table: Table) -> Self {
        Self {
            table,
            snapshot_ids: vec![],
            older_than_ms: None,
            retain_last: None,
            max_concurrent_deletes: DEFAULT_MAX_CONCURRENT_DELETES,
            dry_run: false,
        }
    }

    /// Expire these snapshot ids in addition to any age-based selection.
    pub fn expire_snapshot_ids(mut self, snapshot_ids: impl IntoIterator<Item = i64>) -> Self {
        self.snapshot_ids.extend(snapshot_ids);
        self
    }

    /// Expire snapshots whose timestamp is strictly older than `older_than_ms`.
    pub fn expire_older_than_ms(mut self, older_than_ms: i64) -> Self {
        self.older_than_ms = Some(older_than_ms);
        self
    }

    /// Keep at least the `retain_last` most recent snapshots of each branch when expiring by age.
    pub fn retain_last(mut self, retain_last: usize) -> Self {
        self.retain_last = Some(retain_last);
        self
    }

    /// Bounds how many file deletions run concurrently (default 4, minimum 1).
    pub fn max_concurrent_deletes(mut self, max_concurrent_deletes: usize) -> Self {
        self.max_concurrent_deletes = max_concurrent_deletes;
        self
    }

    /// When set, resolve and report what would be deleted without committing or deleting anything.
    pub fn dry_run(mut self, dry_run: bool) -> Self {
        self.dry_run = dry_run;
        self
    }

    /// Runs the operation: select snapshots, commit the metadata change, then delete files.
    ///
    /// The files to delete are enumerated against the pre-expiry metadata, since the metadata
    /// commit only rewrites metadata and leaves the manifests on storage for this step to read and
    /// then remove. A [`dry_run`](Self::dry_run) returns the same plan without committing or
    /// deleting, and never touches `catalog`.
    pub async fn execute(self, catalog: &dyn Catalog) -> Result<ExpireSnapshotsResult> {
        let tx = Transaction::new(&self.table);
        let mut action = tx.expire_snapshots();
        if !self.snapshot_ids.is_empty() {
            action = action.expire_snapshot_ids(self.snapshot_ids.clone());
        }
        if let Some(older_than_ms) = self.older_than_ms {
            action = action.expire_older_than_ms(older_than_ms);
        }
        if let Some(retain_last) = self.retain_last {
            action = action.retain_last(retain_last);
        }

        let expired_snapshot_ids = action.snapshot_ids_to_expire(&self.table)?;
        let expired: HashSet<i64> = expired_snapshot_ids.iter().copied().collect();
        let files = unreferenced_files(&self.table, &expired).await?;

        if self.dry_run {
            return Ok(ExpireSnapshotsResult::new(
                true,
                expired_snapshot_ids,
                &files,
            ));
        }

        let tx = action.apply(tx)?;
        tx.commit(catalog).await?;

        delete_files(self.table.file_io(), &files, self.max_concurrent_deletes).await?;

        Ok(ExpireSnapshotsResult::new(
            false,
            expired_snapshot_ids,
            &files,
        ))
    }
}

/// Outcome of an [`ExpireSnapshots`] run: the expired snapshots and the per-kind file counts that
/// were deleted (or, for a dry run, that would be deleted).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExpireSnapshotsResult {
    /// Whether this was a dry run (nothing committed or deleted).
    pub dry_run: bool,
    /// Snapshot ids that were (or would be) expired, sorted ascending.
    pub expired_snapshot_ids: Vec<i64>,
    /// Manifest-list files deleted.
    pub manifest_lists: usize,
    /// Manifest files deleted.
    pub manifests: usize,
    /// Data files deleted.
    pub data_files: usize,
    /// Delete files deleted.
    pub delete_files: usize,
    /// Statistics files deleted.
    pub statistics_files: usize,
    /// Partition-statistics files deleted.
    pub partition_statistics_files: usize,
}

impl ExpireSnapshotsResult {
    fn new(dry_run: bool, expired_snapshot_ids: Vec<i64>, files: &UnreferencedFiles) -> Self {
        Self {
            dry_run,
            expired_snapshot_ids,
            manifest_lists: files.manifest_lists.len(),
            manifests: files.manifests.len(),
            data_files: files.data_files.len(),
            delete_files: files.delete_files.len(),
            statistics_files: files.statistics_files.len(),
            partition_statistics_files: files.partition_statistics_files.len(),
        }
    }
}

/// Deletes every file in `files` with up to `max_concurrent` deletions in flight.
async fn delete_files(io: &FileIO, files: &UnreferencedFiles, max_concurrent: usize) -> Result<()> {
    let paths: Vec<String> = files.all_paths().cloned().collect();
    futures::stream::iter(paths)
        .map(|path| async move { io.delete(path).await })
        .buffer_unordered(max_concurrent.max(1))
        .try_collect::<Vec<()>>()
        .await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::sync::Arc;

    use iceberg::io::LocalFsStorageFactory;
    use iceberg::memory::{MEMORY_CATALOG_WAREHOUSE, MemoryCatalogBuilder};
    use iceberg::spec::{
        DataContentType, DataFileBuilder, DataFileFormat, NestedField, PrimitiveType, Schema,
        Struct, Type,
    };
    use iceberg::{
        Catalog, CatalogBuilder, MemoryCatalog, NamespaceIdent, TableCreation, TableIdent,
    };
    use tempfile::TempDir;

    use super::*;

    fn schema() -> Schema {
        Schema::builder()
            .with_schema_id(0)
            .with_fields(vec![
                NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int)).into(),
            ])
            .build()
            .unwrap()
    }

    /// Appends one (synthetic) data file as a new snapshot and returns the updated table.
    async fn append(catalog: &MemoryCatalog, table: Table, name: &str) -> Table {
        let data_file = DataFileBuilder::default()
            .partition_spec_id(0)
            .content(DataContentType::Data)
            .file_path(format!(
                "{}/data/{name}.parquet",
                table.metadata().location()
            ))
            .file_format(DataFileFormat::Parquet)
            .file_size_in_bytes(10)
            .record_count(1)
            .partition(Struct::empty())
            .build()
            .unwrap();
        let tx = Transaction::new(&table);
        let tx = tx
            .fast_append()
            .add_data_files(vec![data_file])
            .apply(tx)
            .unwrap();
        tx.commit(catalog).await.unwrap()
    }

    /// A two-snapshot table (S1 then S2 as main head) registered in a `MemoryCatalog`.
    struct Fixture {
        _tmp: TempDir,
        catalog: MemoryCatalog,
        ident: TableIdent,
        table: Table,
        s1: i64,
        s1_manifest_list: String,
    }

    async fn fixture() -> Fixture {
        let tmp = TempDir::new().unwrap();
        let warehouse = tmp.path().to_str().unwrap().to_string();
        let catalog = MemoryCatalogBuilder::default()
            .with_storage_factory(Arc::new(LocalFsStorageFactory))
            .load(
                "memory",
                HashMap::from([(MEMORY_CATALOG_WAREHOUSE.to_string(), warehouse)]),
            )
            .await
            .unwrap();

        let namespace = NamespaceIdent::new("ns".to_string());
        catalog
            .create_namespace(&namespace, HashMap::new())
            .await
            .unwrap();
        let table = catalog
            .create_table(
                &namespace,
                TableCreation::builder()
                    .name("t".to_string())
                    .schema(schema())
                    .build(),
            )
            .await
            .unwrap();
        let ident = TableIdent::new(namespace, "t".to_string());

        let table = append(&catalog, table, "f1").await;
        let s1 = table.metadata().current_snapshot_id().unwrap();
        let s1_manifest_list = table
            .metadata()
            .snapshot_by_id(s1)
            .unwrap()
            .manifest_list()
            .to_string();
        let table = append(&catalog, table, "f2").await;

        Fixture {
            _tmp: tmp,
            catalog,
            ident,
            table,
            s1,
            s1_manifest_list,
        }
    }

    #[tokio::test]
    async fn execute_expires_snapshot_and_deletes_orphaned_files() {
        let f = fixture().await;
        let io = f.table.file_io().clone();
        assert!(io.exists(&f.s1_manifest_list).await.unwrap());

        // Pin age expiry off so only the explicitly named S1 is expired.
        let result = ExpireSnapshots::new(f.table)
            .expire_snapshot_ids([f.s1])
            .expire_older_than_ms(1)
            .execute(&f.catalog)
            .await
            .unwrap();

        assert!(!result.dry_run);
        assert_eq!(result.expired_snapshot_ids, vec![f.s1]);
        // S1's manifest list is orphaned, but its manifest and data file are still carried by S2.
        assert_eq!(result.manifest_lists, 1);
        assert_eq!(result.manifests, 0);
        assert_eq!(result.data_files, 0);

        assert!(!io.exists(&f.s1_manifest_list).await.unwrap());
        let reloaded = f.catalog.load_table(&f.ident).await.unwrap();
        assert!(reloaded.metadata().snapshot_by_id(f.s1).is_none());
    }

    #[tokio::test]
    async fn dry_run_reports_plan_without_committing_or_deleting() {
        let f = fixture().await;
        let io = f.table.file_io().clone();

        let result = ExpireSnapshots::new(f.table)
            .expire_snapshot_ids([f.s1])
            .expire_older_than_ms(1)
            .dry_run(true)
            .execute(&f.catalog)
            .await
            .unwrap();

        assert!(result.dry_run);
        assert_eq!(result.expired_snapshot_ids, vec![f.s1]);
        assert_eq!(result.manifest_lists, 1);

        // Nothing was deleted or committed.
        assert!(io.exists(&f.s1_manifest_list).await.unwrap());
        let reloaded = f.catalog.load_table(&f.ident).await.unwrap();
        assert!(reloaded.metadata().snapshot_by_id(f.s1).is_some());
    }

    #[tokio::test]
    async fn no_op_when_nothing_expires() {
        let f = fixture().await;
        let result = ExpireSnapshots::new(f.table)
            .retain_last(10)
            .expire_older_than_ms(1)
            .execute(&f.catalog)
            .await
            .unwrap();
        assert!(result.expired_snapshot_ids.is_empty());
        assert_eq!(result.manifest_lists, 0);
    }
}
