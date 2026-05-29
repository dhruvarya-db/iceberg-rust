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
use std::sync::Arc;

use async_trait::async_trait;

use crate::table::Table;
use crate::transaction::action::{ActionCommit, TransactionAction};
use crate::{Error, ErrorKind, Result, TableRequirement, TableUpdate};

/// Default number of most recent snapshots to retain when none is specified.
const DEFAULT_RETAIN_LAST: usize = 1;

/// A transaction action that removes snapshots from table metadata.
///
/// This only rewrites metadata; the now-unreferenced data and metadata files are left untouched.
/// Physical file cleanup is the responsibility of a higher-level maintenance operation built on
/// top of this action.
///
/// Selection follows the Spark `expire_snapshots` semantics:
/// - When explicit ids are set via [`expire_snapshot_ids`](Self::expire_snapshot_ids), exactly
///   those snapshots are expired and the age/`retain_last` filters are ignored.
/// - Otherwise, snapshots older than [`expire_older_than_ms`](Self::expire_older_than_ms) are
///   expired while always keeping the most recent
///   [`retain_last`](Self::retain_last) snapshots.
///
/// The current snapshot is never expired.
pub struct ExpireSnapshotsAction {
    snapshot_ids: Vec<i64>,
    older_than_ms: Option<i64>,
    retain_last: Option<usize>,
}

impl ExpireSnapshotsAction {
    pub(crate) fn new() -> Self {
        Self {
            snapshot_ids: vec![],
            older_than_ms: None,
            retain_last: None,
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

    /// Resolves the snapshot ids this action would remove from the given table.
    ///
    /// Exposed so maintenance operations can compute the files those snapshots leave behind
    /// without duplicating the selection rules.
    pub fn snapshot_ids_to_expire(&self, table: &Table) -> Result<Vec<i64>> {
        let metadata = table.metadata();
        let current_snapshot_id = metadata.current_snapshot_id();

        if !self.snapshot_ids.is_empty() {
            if current_snapshot_id.is_some_and(|current| self.snapshot_ids.contains(&current)) {
                return Err(Error::new(
                    ErrorKind::DataInvalid,
                    "Cannot expire the current snapshot",
                ));
            }
            let existing: HashSet<i64> = metadata.snapshots().map(|s| s.snapshot_id()).collect();
            return Ok(self
                .snapshot_ids
                .iter()
                .copied()
                .filter(|id| existing.contains(id))
                .collect());
        }

        let retain_last = self.retain_last.unwrap_or(DEFAULT_RETAIN_LAST);
        let mut snapshots: Vec<_> = metadata.snapshots().cloned().collect();
        if snapshots.len() <= retain_last {
            return Ok(vec![]);
        }
        snapshots.sort_by_key(|s| s.timestamp_ms());

        // Drop the most recent `retain_last` snapshots from the expiry candidates.
        snapshots.truncate(snapshots.len() - retain_last);

        Ok(snapshots
            .into_iter()
            .filter(|s| self.older_than_ms.is_none_or(|t| s.timestamp_ms() < t))
            .filter(|s| current_snapshot_id != Some(s.snapshot_id()))
            .map(|s| s.snapshot_id())
            .collect())
    }
}

#[async_trait]
impl TransactionAction for ExpireSnapshotsAction {
    async fn commit(self: Arc<Self>, table: &Table) -> Result<ActionCommit> {
        let snapshot_ids = self.snapshot_ids_to_expire(table)?;

        if snapshot_ids.is_empty() {
            return Ok(ActionCommit::new(vec![], vec![]));
        }

        let metadata = table.metadata();
        let mut updates = vec![TableUpdate::RemoveSnapshots {
            snapshot_ids: snapshot_ids.clone(),
        }];

        // Drop statistics tied to expired snapshots so metadata holds no dangling references.
        for snapshot_id in &snapshot_ids {
            if metadata.statistics_for_snapshot(*snapshot_id).is_some() {
                updates.push(TableUpdate::RemoveStatistics {
                    snapshot_id: *snapshot_id,
                });
            }
            if metadata
                .partition_statistics_for_snapshot(*snapshot_id)
                .is_some()
            {
                updates.push(TableUpdate::RemovePartitionStatistics {
                    snapshot_id: *snapshot_id,
                });
            }
        }

        Ok(ActionCommit::new(updates, vec![
            TableRequirement::UuidMatch {
                uuid: metadata.uuid(),
            },
        ]))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use crate::TableUpdate;
    use crate::transaction::Transaction;
    use crate::transaction::action::{ApplyTransactionAction, TransactionAction};
    use crate::transaction::expire_snapshots::ExpireSnapshotsAction;
    use crate::transaction::tests::make_v2_table;

    // `make_v2_table` carries an older snapshot (ts 1515100955770) and a current
    // snapshot (ts 1555100955770).
    const OLD_SNAPSHOT: i64 = 3051729675574597004;
    const CURRENT_SNAPSHOT: i64 = 3055729675574597004;

    async fn removed_ids(action: ExpireSnapshotsAction) -> Vec<i64> {
        let table = make_v2_table();
        let mut commit = Arc::new(action).commit(&table).await.unwrap();
        match commit.take_updates().into_iter().next() {
            Some(TableUpdate::RemoveSnapshots { snapshot_ids }) => snapshot_ids,
            None => vec![],
            other => panic!("unexpected update: {other:?}"),
        }
    }

    fn action() -> ExpireSnapshotsAction {
        ExpireSnapshotsAction::new()
    }

    #[tokio::test]
    async fn test_expire_explicit_snapshot_id() {
        assert_eq!(
            removed_ids(action().expire_snapshot_ids(vec![OLD_SNAPSHOT])).await,
            vec![OLD_SNAPSHOT]
        );
    }

    #[tokio::test]
    async fn test_explicit_unknown_id_is_ignored() {
        assert!(
            removed_ids(action().expire_snapshot_ids(vec![42]))
                .await
                .is_empty()
        );
    }

    #[tokio::test]
    async fn test_cannot_expire_current_snapshot() {
        let table = make_v2_table();
        let action = action().expire_snapshot_ids(vec![CURRENT_SNAPSHOT]);
        assert!(Arc::new(action).commit(&table).await.is_err());
    }

    #[tokio::test]
    async fn test_retain_last_default_expires_older_non_current() {
        assert_eq!(
            removed_ids(action().expire_older_than_ms(i64::MAX)).await,
            vec![OLD_SNAPSHOT]
        );
    }

    #[tokio::test]
    async fn test_retain_last_noop_when_enough_retained() {
        assert!(removed_ids(action().retain_last(5)).await.is_empty());
    }

    #[tokio::test]
    async fn test_expire_removes_statistics_of_expired_snapshot() {
        use std::sync::Arc;

        use crate::spec::StatisticsFile;

        let table = make_v2_table();
        let metadata = table
            .metadata()
            .clone()
            .into_builder(None)
            .set_statistics(StatisticsFile {
                snapshot_id: OLD_SNAPSHOT,
                statistics_path: "stats/old.puffin".to_string(),
                file_size_in_bytes: 1,
                file_footer_size_in_bytes: 1,
                key_metadata: None,
                blob_metadata: vec![],
            })
            .build()
            .unwrap()
            .metadata;
        let table = table.with_metadata(Arc::new(metadata));

        let mut commit = Arc::new(action().expire_snapshot_ids(vec![OLD_SNAPSHOT]))
            .commit(&table)
            .await
            .unwrap();

        assert!(
            commit
                .take_updates()
                .contains(&TableUpdate::RemoveStatistics {
                    snapshot_id: OLD_SNAPSHOT,
                })
        );
    }

    #[tokio::test]
    async fn test_older_than_excludes_newer_snapshots() {
        // Threshold older than every snapshot -> nothing qualifies.
        assert!(
            removed_ids(action().expire_older_than_ms(1))
                .await
                .is_empty()
        );
    }

    #[tokio::test]
    async fn test_apply_registers_action() {
        let table = make_v2_table();
        let tx = Transaction::new(&table);
        let tx = tx
            .expire_snapshots()
            .expire_snapshot_ids(vec![OLD_SNAPSHOT])
            .apply(tx)
            .unwrap();
        assert_eq!(tx.actions.len(), 1);
    }
}
