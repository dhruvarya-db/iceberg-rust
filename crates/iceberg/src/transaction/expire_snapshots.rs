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

use crate::spec::{MAIN_BRANCH, SnapshotRetention, TableMetadata};
use crate::table::Table;
use crate::transaction::action::{ActionCommit, TransactionAction};
use crate::{Error, ErrorKind, Result, TableRequirement, TableUpdate};

/// Default number of most recent snapshots to retain per branch when none is specified.
const DEFAULT_RETAIN_LAST: usize = 1;

/// A transaction action that removes snapshots from table metadata.
///
/// This only rewrites metadata; the now-unreferenced data and metadata files are left untouched.
/// Physical file cleanup is the responsibility of a higher-level maintenance operation built on
/// top of this action.
///
/// Selection follows Java `RemoveSnapshots`:
/// - Explicit ids ([`expire_snapshot_ids`](Self::expire_snapshot_ids)) and age-based expiry
///   ([`expire_older_than_ms`](Self::expire_older_than_ms)) are combined: a snapshot is expired
///   if it is named explicitly *or* it is selected by age.
/// - Age-based expiry is computed per branch along each branch's ancestry: each branch keeps its
///   most recent [`retain_last`](Self::retain_last) snapshots (a per-ref `min_snapshots_to_keep`
///   overrides it) plus any snapshot newer than the cutoff, so a shared ancestor reachable from a
///   retained branch is never expired.
/// - Ref heads (branch heads and tag targets, including the current snapshot) are never expired,
///   and naming one explicitly is an error, since
///   [`remove_snapshots`](crate::spec::TableMetadataBuilder::remove_snapshots) would otherwise
///   drop the ref silently.
///
/// The action's single [`expire_older_than_ms`](Self::expire_older_than_ms) cutoff is used for
/// every branch; per-ref age windows (`max_snapshot_age_ms`, `max_ref_age_ms`) and ref aging are
/// not applied.
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

    /// Expire these snapshot ids in addition to any age-based selection.
    ///
    /// Ids accumulate across calls (like [`add_data_files`](crate::transaction::Transaction::fast_append)).
    /// An id that is still referenced by a branch or tag cannot be expired and causes
    /// [`commit`](TransactionAction::commit) to fail.
    pub fn expire_snapshot_ids(mut self, snapshot_ids: impl IntoIterator<Item = i64>) -> Self {
        self.snapshot_ids.extend(snapshot_ids);
        self
    }

    /// Expire snapshots whose timestamp is strictly older than `older_than_ms`.
    pub fn expire_older_than_ms(mut self, older_than_ms: i64) -> Self {
        self.older_than_ms = Some(older_than_ms);
        self
    }

    /// Keep at least the `retain_last` most recent snapshots of each branch when expiring by age
    /// (defaults to 1).
    ///
    /// This only bounds [`expire_older_than_ms`](Self::expire_older_than_ms); it has no effect on
    /// its own and does not protect snapshots named via
    /// [`expire_snapshot_ids`](Self::expire_snapshot_ids).
    pub fn retain_last(mut self, retain_last: usize) -> Self {
        self.retain_last = Some(retain_last);
        self
    }

    fn snapshot_ids_to_expire(&self, table: &Table) -> Result<Vec<i64>> {
        let metadata = table.metadata();
        let ref_heads = Self::ref_head_ids(metadata);
        let existing: HashSet<i64> = metadata.snapshots().map(|s| s.snapshot_id()).collect();

        let mut to_expire: HashSet<i64> = HashSet::new();

        // Explicit ids are expired regardless of age, but one still referenced by a branch or tag
        // cannot be expired (Java's RemoveSnapshots errors rather than silently dropping the ref).
        for id in &self.snapshot_ids {
            if ref_heads.contains(id) {
                return Err(Self::reference_error(metadata, *id));
            }
            if existing.contains(id) {
                to_expire.insert(*id);
            }
        }

        // Age-based expiry: expire any snapshot not retained by per-branch retention. Without a
        // cutoff there is no age expiry.
        if let Some(cutoff) = self.older_than_ms {
            let retained = self.snapshot_ids_to_retain(metadata, cutoff);
            for snapshot in metadata.snapshots() {
                if !retained.contains(&snapshot.snapshot_id()) {
                    to_expire.insert(snapshot.snapshot_id());
                }
            }
        }

        let mut snapshot_ids: Vec<i64> = to_expire.into_iter().collect();
        snapshot_ids.sort_unstable();
        Ok(snapshot_ids)
    }

    /// Ref heads (branch heads and tag targets) plus the current snapshot. These can never be
    /// expired; naming one explicitly is an error, since
    /// [`remove_snapshots`](crate::spec::TableMetadataBuilder::remove_snapshots) would otherwise
    /// drop the ref silently.
    fn ref_head_ids(metadata: &TableMetadata) -> HashSet<i64> {
        let mut ids: HashSet<i64> = metadata.refs.values().map(|r| r.snapshot_id).collect();
        if let Some(current) = metadata.current_snapshot_id() {
            ids.insert(current);
        }
        ids
    }

    /// Snapshots retained by age-based expiry, mirroring Java `RemoveSnapshots`: every ref head,
    /// the most recent `retain_last` ancestors of each branch (its floor), and any snapshot newer
    /// than the cutoff (covering young ancestors and unreferenced-but-young snapshots).
    fn snapshot_ids_to_retain(&self, metadata: &TableMetadata, cutoff: i64) -> HashSet<i64> {
        let retain_last = self.retain_last.unwrap_or(DEFAULT_RETAIN_LAST);
        let mut retained = Self::ref_head_ids(metadata);

        // Per-branch floor: keep the newest `min` ancestors of each branch head. The current
        // snapshot is always treated as a branch head so its lineage is protected even when the
        // metadata has no explicit `main` ref.
        let mut branch_heads: Vec<(i64, usize)> = metadata
            .refs
            .values()
            .filter_map(|r| match &r.retention {
                SnapshotRetention::Branch {
                    min_snapshots_to_keep,
                    ..
                } => Some((
                    r.snapshot_id,
                    min_snapshots_to_keep.map_or(retain_last, |m| m as usize),
                )),
                SnapshotRetention::Tag { .. } => None,
            })
            .collect();
        if let Some(current) = metadata.current_snapshot_id() {
            branch_heads.push((current, retain_last));
        }
        for (head, min) in branch_heads {
            retained.extend(Self::ancestors(metadata, head).take(min));
        }

        // Anything newer than the cutoff is retained regardless of reachability.
        for snapshot in metadata.snapshots() {
            if snapshot.timestamp_ms() >= cutoff {
                retained.insert(snapshot.snapshot_id());
            }
        }

        retained
    }

    /// Iterates a snapshot and its ancestors, newest first, following `parent_snapshot_id`.
    fn ancestors(metadata: &TableMetadata, head: i64) -> impl Iterator<Item = i64> + '_ {
        let mut next = Some(head);
        std::iter::from_fn(move || {
            let id = next?;
            next = metadata
                .snapshot_by_id(id)
                .and_then(|snapshot| snapshot.parent_snapshot_id());
            Some(id)
        })
    }

    fn reference_error(metadata: &TableMetadata, snapshot_id: i64) -> Error {
        if metadata.current_snapshot_id() == Some(snapshot_id) {
            return Error::new(ErrorKind::DataInvalid, "Cannot expire the current snapshot");
        }
        let refs: Vec<&str> = metadata
            .refs
            .iter()
            .filter(|(_, r)| r.snapshot_id == snapshot_id)
            .map(|(name, _)| name.as_str())
            .collect();
        Error::new(
            ErrorKind::DataInvalid,
            format!("Cannot expire snapshot {snapshot_id}: still referenced by {refs:?}"),
        )
    }
}

#[async_trait]
impl TransactionAction for ExpireSnapshotsAction {
    async fn commit(self: Arc<Self>, table: &Table) -> Result<ActionCommit> {
        let metadata = table.metadata();

        // Expiring metadata defeats a user's explicit decision to disable GC (Java refuses too).
        if !metadata.table_properties()?.gc_enabled {
            return Err(Error::new(
                ErrorKind::DataInvalid,
                "Cannot expire snapshots: gc.enabled is false",
            ));
        }

        let snapshot_ids = self.snapshot_ids_to_expire(table)?;

        if snapshot_ids.is_empty() {
            return Ok(ActionCommit::new(vec![], vec![]));
        }

        // The ref assertion closes the race where a concurrent writer advances `main` between
        // selection and commit, which could orphan a snapshot whose parent we are about to remove.
        Ok(ActionCommit::new(
            vec![TableUpdate::RemoveSnapshots { snapshot_ids }],
            vec![
                TableRequirement::UuidMatch {
                    uuid: metadata.uuid(),
                },
                TableRequirement::RefSnapshotIdMatch {
                    r#ref: MAIN_BRANCH.to_string(),
                    snapshot_id: metadata.current_snapshot_id(),
                },
            ],
        ))
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::sync::Arc;

    use crate::spec::{
        MAIN_BRANCH, Operation, Snapshot, SnapshotReference, SnapshotRetention, Summary,
    };
    use crate::table::Table;
    use crate::transaction::Transaction;
    use crate::transaction::action::{ApplyTransactionAction, TransactionAction};
    use crate::transaction::expire_snapshots::ExpireSnapshotsAction;
    use crate::transaction::tests::{make_v2_minimal_table, make_v2_table};
    use crate::{TableRequirement, TableUpdate};

    // `make_v2_table` carries an older snapshot (ts 1515100955770) and a current
    // snapshot (ts 1555100955770).
    const OLD_SNAPSHOT: i64 = 3051729675574597004;
    const CURRENT_SNAPSHOT: i64 = 3055729675574597004;
    // Well after the minimal table's last-updated-ms, so synthetic snapshots pass timestamp checks.
    const TS: i64 = 1_700_000_000_000;

    fn action() -> ExpireSnapshotsAction {
        ExpireSnapshotsAction::new()
    }

    async fn removed_ids(action: ExpireSnapshotsAction) -> Vec<i64> {
        expired(&make_v2_table(), action).await
    }

    async fn expired(table: &Table, action: ExpireSnapshotsAction) -> Vec<i64> {
        let mut commit = Arc::new(action).commit(table).await.unwrap();
        match commit.take_updates().into_iter().next() {
            Some(TableUpdate::RemoveSnapshots { snapshot_ids }) => snapshot_ids,
            None => vec![],
            other => panic!("unexpected update: {other:?}"),
        }
    }

    fn snapshot(id: i64, parent: Option<i64>, sequence_number: i64, timestamp_ms: i64) -> Snapshot {
        Snapshot::builder()
            .with_snapshot_id(id)
            .with_parent_snapshot_id(parent)
            .with_sequence_number(sequence_number)
            .with_timestamp_ms(timestamp_ms)
            .with_schema_id(0)
            .with_manifest_list(format!("/snap-{id}.avro"))
            .with_summary(Summary {
                operation: Operation::Append,
                additional_properties: HashMap::new(),
            })
            .build()
    }

    fn branch(snapshot_id: i64, min_snapshots_to_keep: Option<i32>) -> SnapshotReference {
        SnapshotReference {
            snapshot_id,
            retention: SnapshotRetention::Branch {
                min_snapshots_to_keep,
                max_snapshot_age_ms: None,
                max_ref_age_ms: None,
            },
        }
    }

    /// Builds a table from synthetic snapshots and refs on top of an empty base.
    fn table_with(snapshots: Vec<Snapshot>, refs: Vec<(&str, SnapshotReference)>) -> Table {
        let base = make_v2_minimal_table();
        let mut builder = base.metadata().clone().into_builder(None);
        for snapshot in snapshots {
            builder = builder.add_snapshot(snapshot).unwrap();
        }
        for (name, reference) in refs {
            builder = builder.set_ref(name, reference).unwrap();
        }
        base.with_metadata(Arc::new(builder.build().unwrap().metadata))
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

    /// `make_v2_table` with a tag pointing at the older snapshot.
    fn table_with_tag_on_old() -> Table {
        let table = make_v2_table();
        let metadata = table
            .metadata()
            .clone()
            .into_builder(None)
            .set_ref("history-tag", SnapshotReference {
                snapshot_id: OLD_SNAPSHOT,
                retention: SnapshotRetention::Tag {
                    max_ref_age_ms: None,
                },
            })
            .unwrap()
            .build()
            .unwrap()
            .metadata;
        table.with_metadata(Arc::new(metadata))
    }

    #[tokio::test]
    async fn test_cannot_expire_tagged_snapshot_explicitly() {
        let table = table_with_tag_on_old();
        let action = action().expire_snapshot_ids(vec![OLD_SNAPSHOT]);
        assert!(Arc::new(action).commit(&table).await.is_err());
    }

    #[tokio::test]
    async fn test_age_expiry_skips_tagged_snapshot() {
        let table = table_with_tag_on_old();
        let mut commit = Arc::new(action().expire_older_than_ms(i64::MAX))
            .commit(&table)
            .await
            .unwrap();
        // Both snapshots are referenced (current + tag), so nothing is expired.
        assert!(commit.take_updates().is_empty());
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

    #[tokio::test]
    async fn test_per_branch_retention_protects_shared_ancestor() {
        // main: 1 -> 2 -> 3 ; branch `b`: 1 -> 2 -> 4. Snapshot 2 is a shared ancestor of both
        // branches but is not a ref head.
        let table = table_with(
            vec![
                snapshot(1, None, 35, TS + 1),
                snapshot(2, Some(1), 36, TS + 2),
                snapshot(3, Some(2), 37, TS + 3),
                snapshot(4, Some(2), 38, TS + 4),
            ],
            vec![(MAIN_BRANCH, branch(3, None)), ("b", branch(4, None))],
        );

        // A global "newest 2" would expire 2 and orphan branch `b`; per-branch retention keeps it.
        let removed = expired(
            &table,
            action().retain_last(2).expire_older_than_ms(i64::MAX),
        )
        .await;
        assert_eq!(removed, vec![1]);
    }

    #[tokio::test]
    async fn test_per_ref_min_snapshots_to_keep_overrides_retain_last() {
        let table = table_with(
            vec![
                snapshot(1, None, 35, TS + 1),
                snapshot(2, Some(1), 36, TS + 2),
                snapshot(3, Some(2), 37, TS + 3),
            ],
            vec![(MAIN_BRANCH, branch(3, Some(3)))],
        );

        // The branch's own min_snapshots_to_keep=3 wins over the action's retain_last(1).
        let removed = expired(
            &table,
            action().retain_last(1).expire_older_than_ms(i64::MAX),
        )
        .await;
        assert!(removed.is_empty());
    }

    #[tokio::test]
    async fn test_explicit_and_age_combine() {
        let table = table_with(
            vec![
                snapshot(1, None, 35, TS + 1),
                snapshot(2, Some(1), 36, TS + 2),
                snapshot(3, Some(2), 37, TS + 3),
                snapshot(4, Some(3), 38, TS + 4),
            ],
            vec![(MAIN_BRANCH, branch(4, None))],
        );

        // Age expires 1 and 2 (older than the cutoff, beyond retain_last). 3 is newer than the
        // cutoff so age keeps it, but it is named explicitly, so all three are expired.
        let removed = expired(
            &table,
            action()
                .retain_last(1)
                .expire_older_than_ms(TS + 3)
                .expire_snapshot_ids(vec![3]),
        )
        .await;
        assert_eq!(removed, vec![1, 2, 3]);
    }

    #[tokio::test]
    async fn test_expire_snapshot_ids_accumulates() {
        let table = table_with(
            vec![
                snapshot(1, None, 35, TS + 1),
                snapshot(2, Some(1), 36, TS + 2),
                snapshot(3, Some(2), 37, TS + 3),
            ],
            vec![(MAIN_BRANCH, branch(3, None))],
        );

        // Two separate calls both take effect.
        let removed = expired(
            &table,
            action()
                .expire_snapshot_ids(vec![1])
                .expire_snapshot_ids(vec![2]),
        )
        .await;
        assert_eq!(removed, vec![1, 2]);
    }

    #[tokio::test]
    async fn test_gc_disabled_errors() {
        let table = make_v2_table();
        let metadata = table
            .metadata()
            .clone()
            .into_builder(None)
            .set_properties(HashMap::from([(
                "gc.enabled".to_string(),
                "false".to_string(),
            )]))
            .unwrap()
            .build()
            .unwrap()
            .metadata;
        let table = table.with_metadata(Arc::new(metadata));

        let action = action().expire_snapshot_ids(vec![OLD_SNAPSHOT]);
        assert!(Arc::new(action).commit(&table).await.is_err());
    }

    #[tokio::test]
    async fn test_commit_asserts_main_ref() {
        let table = make_v2_table();
        let mut commit = Arc::new(action().expire_snapshot_ids(vec![OLD_SNAPSHOT]))
            .commit(&table)
            .await
            .unwrap();
        assert!(
            commit
                .take_requirements()
                .iter()
                .any(|requirement| matches!(
                    requirement,
                    TableRequirement::RefSnapshotIdMatch { r#ref, snapshot_id }
                        if r#ref == MAIN_BRANCH && *snapshot_id == Some(CURRENT_SNAPSHOT)
                ))
        );
    }
}
