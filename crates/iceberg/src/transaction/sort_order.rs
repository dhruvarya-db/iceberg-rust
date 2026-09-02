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

use std::sync::Arc;

use async_trait::async_trait;

use crate::error::Result;
use crate::spec::{NullOrder, SchemaRef, SortDirection, SortField, SortOrder, Transform};
use crate::table::Table;
use crate::transaction::{ActionCommit, TransactionAction};
use crate::{Error, ErrorKind, TableRequirement, TableUpdate};

/// Represents a sort field whose construction and validation are deferred until commit time.
/// This avoids the need to pass a `Table` reference into methods like `asc`, `desc`, or
/// `sort_by` when adding sort orders.
#[derive(Debug, PartialEq, Eq, Clone)]
struct PendingSortField {
    name: String,
    transform: Transform,
    direction: SortDirection,
    null_order: NullOrder,
}

impl PendingSortField {
    fn to_sort_field(&self, schema: &SchemaRef) -> Result<SortField> {
        let field_id = schema.field_id_by_name(self.name.as_str()).ok_or_else(|| {
            Error::new(
                ErrorKind::DataInvalid,
                format!("Cannot find field {} in table schema", self.name),
            )
        })?;

        Ok(SortField::builder()
            .source_id(field_id)
            .transform(self.transform)
            .direction(self.direction)
            .null_order(self.null_order)
            .build())
    }
}

/// Transaction action for replacing sort order.
pub struct ReplaceSortOrderAction {
    pending_sort_fields: Vec<PendingSortField>,
}

impl ReplaceSortOrderAction {
    pub(crate) fn new() -> Self {
        ReplaceSortOrderAction {
            pending_sort_fields: vec![],
        }
    }

    /// Adds a field for sorting in ascending order on the raw column value.
    pub fn asc(self, name: &str, null_order: NullOrder) -> Self {
        self.sort_by(
            name,
            Transform::Identity,
            SortDirection::Ascending,
            null_order,
        )
    }

    /// Adds a field for sorting in descending order on the raw column value.
    pub fn desc(self, name: &str, null_order: NullOrder) -> Self {
        self.sort_by(
            name,
            Transform::Identity,
            SortDirection::Descending,
            null_order,
        )
    }

    /// Adds a field for sorting on the result of `transform` applied to the named column,
    /// for example [`Transform::Bucket`] or [`Transform::Truncate`]. Pass
    /// [`Transform::Identity`] to sort on the raw value (equivalent to [`Self::asc`] /
    /// [`Self::desc`]).
    ///
    /// The column and transform are resolved against the table schema when the transaction
    /// commits, not here: a missing column, or a transform that is incompatible with the
    /// column's type, fails at commit time.
    pub fn sort_by(
        mut self,
        name: &str,
        transform: Transform,
        direction: SortDirection,
        null_order: NullOrder,
    ) -> Self {
        self.pending_sort_fields.push(PendingSortField {
            name: name.to_string(),
            transform,
            direction,
            null_order,
        });

        self
    }
}

#[async_trait]
impl TransactionAction for ReplaceSortOrderAction {
    async fn commit(self: Arc<Self>, table: &Table) -> Result<ActionCommit> {
        let current_schema = table.metadata().current_schema();
        let sort_fields: Result<Vec<SortField>> = self
            .pending_sort_fields
            .iter()
            .map(|p| p.to_sort_field(current_schema))
            .collect();

        let bound_sort_order = SortOrder::builder()
            .with_fields(sort_fields?)
            .build(current_schema)?;

        let updates = vec![
            TableUpdate::AddSortOrder {
                sort_order: bound_sort_order,
            },
            TableUpdate::SetDefaultSortOrder { sort_order_id: -1 },
        ];

        let requirements = vec![
            TableRequirement::CurrentSchemaIdMatch {
                current_schema_id: current_schema.schema_id(),
            },
            TableRequirement::DefaultSortOrderIdMatch {
                default_sort_order_id: table.metadata().default_sort_order().order_id,
            },
        ];

        Ok(ActionCommit::new(updates, requirements))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use as_any::Downcast;

    use crate::TableUpdate;
    use crate::spec::{NullOrder, SortDirection, Transform};
    use crate::transaction::sort_order::{PendingSortField, ReplaceSortOrderAction};
    use crate::transaction::tests::make_v2_table;
    use crate::transaction::{ApplyTransactionAction, Transaction, TransactionAction};

    #[test]
    fn test_replace_sort_order() {
        let table = make_v2_table();
        let tx = Transaction::new(&table);
        let replace_sort_order = tx.replace_sort_order();

        let tx = replace_sort_order
            .asc("x", NullOrder::First)
            .desc("y", NullOrder::Last)
            .apply(tx)
            .unwrap();

        let replace_sort_order = (*tx.actions[0])
            .downcast_ref::<ReplaceSortOrderAction>()
            .unwrap();

        assert_eq!(replace_sort_order.pending_sort_fields, vec![
            PendingSortField {
                name: String::from("x"),
                transform: Transform::Identity,
                direction: SortDirection::Ascending,
                null_order: NullOrder::First,
            },
            PendingSortField {
                name: String::from("y"),
                transform: Transform::Identity,
                direction: SortDirection::Descending,
                null_order: NullOrder::Last,
            }
        ]);
    }

    #[test]
    fn test_sort_by_records_transform() {
        let table = make_v2_table();
        let tx = Transaction::new(&table);
        let replace_sort_order = tx.replace_sort_order();

        let tx = replace_sort_order
            .sort_by(
                "y",
                Transform::Bucket(4),
                SortDirection::Descending,
                NullOrder::Last,
            )
            .sort_by(
                "z",
                Transform::Truncate(10),
                SortDirection::Ascending,
                NullOrder::First,
            )
            .apply(tx)
            .unwrap();

        let replace_sort_order = (*tx.actions[0])
            .downcast_ref::<ReplaceSortOrderAction>()
            .unwrap();

        assert_eq!(replace_sort_order.pending_sort_fields, vec![
            PendingSortField {
                name: String::from("y"),
                transform: Transform::Bucket(4),
                direction: SortDirection::Descending,
                null_order: NullOrder::Last,
            },
            PendingSortField {
                name: String::from("z"),
                transform: Transform::Truncate(10),
                direction: SortDirection::Ascending,
                null_order: NullOrder::First,
            }
        ]);
    }

    #[tokio::test]
    async fn test_commit_builds_sort_order_with_transform() {
        let table = make_v2_table();
        let action = ReplaceSortOrderAction::new().sort_by(
            "y",
            Transform::Bucket(4),
            SortDirection::Descending,
            NullOrder::Last,
        );

        let mut commit = Arc::new(action).commit(&table).await.unwrap();
        let updates = commit.take_updates();

        let TableUpdate::AddSortOrder { sort_order } = &updates[0] else {
            panic!(
                "expected the first update to be AddSortOrder, got {:?}",
                updates[0]
            );
        };
        assert_eq!(sort_order.fields.len(), 1);
        let field = &sort_order.fields[0];
        assert_eq!(field.transform, Transform::Bucket(4));
        assert_eq!(field.source_id, 2); // "y" in the current schema
        assert_eq!(field.direction, SortDirection::Descending);
        assert_eq!(field.null_order, NullOrder::Last);
    }

    #[tokio::test]
    async fn test_commit_rejects_transform_incompatible_with_column_type() {
        let table = make_v2_table();
        // "x" is a `long` column, so a temporal `year` transform is invalid for it.
        let action = ReplaceSortOrderAction::new().sort_by(
            "x",
            Transform::Year,
            SortDirection::Ascending,
            NullOrder::First,
        );

        let Err(err) = Arc::new(action).commit(&table).await else {
            panic!("expected commit to fail for a transform incompatible with the column type");
        };
        // Ensure the error is the transform/type compatibility check, not another commit error.
        assert!(
            err.message().contains("Invalid source type"),
            "expected a transform-compatibility error, got: {err}"
        );
    }

    #[tokio::test]
    async fn test_commit_asc_uses_identity_transform() {
        let table = make_v2_table();
        let action = ReplaceSortOrderAction::new().asc("x", NullOrder::First);

        let mut commit = Arc::new(action).commit(&table).await.unwrap();
        let updates = commit.take_updates();

        let TableUpdate::AddSortOrder { sort_order } = &updates[0] else {
            panic!(
                "expected the first update to be AddSortOrder, got {:?}",
                updates[0]
            );
        };
        assert_eq!(sort_order.fields.len(), 1);
        let field = &sort_order.fields[0];
        assert_eq!(field.transform, Transform::Identity);
        assert_eq!(field.source_id, 1); // "x" in the current schema
        assert_eq!(field.direction, SortDirection::Ascending);
        assert_eq!(field.null_order, NullOrder::First);
    }
}
