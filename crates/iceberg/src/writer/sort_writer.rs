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

//! This module provides the `SortWriter`, a writer that arranges rows to match a
//! table's sort order before handing them to an inner writer.

use std::mem;

use arrow_array::RecordBatch;
use arrow_ord::sort::{SortColumn, lexsort_to_indices};
use arrow_schema::SortOptions;
use arrow_select::concat::concat_batches;
use arrow_select::take::take;
use async_trait::async_trait;

use crate::arrow::record_batch_projector::RecordBatchProjector;
use crate::spec::{
    DataFile, NullOrder, PartitionKey, SchemaRef, SortDirection, SortOrderRef, Transform,
};
use crate::transform::{BoxedTransformFunction, create_transform_function};
use crate::writer::{IcebergWriter, IcebergWriterBuilder};
use crate::{Error, ErrorKind, Result};

/// Builder for [`SortWriter`].
///
/// Wraps an inner [`IcebergWriterBuilder`] and produces writers that sort their input by
/// `sort_order` before delegating. Validation is layered: constructing the builder resolves each
/// sort field's source column against `schema` and fails if one is missing; whether a field's
/// transform is supported is checked when a writer is built; and whether the transform is
/// compatible with the column type is checked when the buffered data is sorted at close.
///
/// The writer resolves sort-key columns by their position in `schema` once, so every batch passed
/// to the writer must have the same column layout as `schema`.
#[derive(Debug)]
pub struct SortWriterBuilder<B> {
    inner: B,
    projector: RecordBatchProjector,
    sort_specs: Vec<SortSpec>,
}

#[derive(Debug, Clone)]
struct SortSpec {
    transform: Transform,
    options: SortOptions,
}

impl<B> SortWriterBuilder<B> {
    /// Create a new `SortWriterBuilder`.
    ///
    /// `sort_order` is the order to write rows in and `schema` is the table schema its fields
    /// reference. An unsorted `sort_order` produces a pass-through writer that buffers but does
    /// not reorder.
    pub fn new(inner: B, sort_order: SortOrderRef, schema: SchemaRef) -> Result<Self> {
        let source_ids: Vec<i32> = sort_order.fields.iter().map(|f| f.source_id).collect();
        let projector = RecordBatchProjector::from_iceberg_schema(schema, &source_ids)?;
        let sort_specs = sort_order
            .fields
            .iter()
            .map(|field| SortSpec {
                transform: field.transform,
                options: SortOptions {
                    descending: matches!(field.direction, SortDirection::Descending),
                    nulls_first: matches!(field.null_order, NullOrder::First),
                },
            })
            .collect();

        Ok(Self {
            inner,
            projector,
            sort_specs,
        })
    }
}

#[async_trait]
impl<B: IcebergWriterBuilder> IcebergWriterBuilder for SortWriterBuilder<B> {
    type R = SortWriter<B::R>;

    async fn build(&self, partition_key: Option<PartitionKey>) -> Result<Self::R> {
        // Resolve transforms before building the inner writer so an unsupported transform
        // (e.g. `Transform::Unknown`) fails without constructing a writer we would discard.
        let transforms = self
            .sort_specs
            .iter()
            .map(|spec| create_transform_function(&spec.transform))
            .collect::<Result<Vec<_>>>()?;
        let options = self.sort_specs.iter().map(|spec| spec.options).collect();
        let inner = self.inner.build(partition_key).await?;

        Ok(SortWriter {
            inner,
            buffer: Vec::new(),
            projector: self.projector.clone(),
            transforms,
            options,
        })
    }
}

/// A writer that buffers its input and, on close, reorders it to match a sort order before
/// writing it to the inner writer.
///
/// Producing a correctly sorted result requires all rows up front, so the writer holds its input
/// in memory until [`close`](IcebergWriter::close). It performs a local sort of the rows it
/// receives; it does not sort across sibling writers (e.g. per-partition writers each sort their
/// own rows).
pub struct SortWriter<W> {
    inner: W,
    buffer: Vec<RecordBatch>,
    projector: RecordBatchProjector,
    transforms: Vec<BoxedTransformFunction>,
    options: Vec<SortOptions>,
}

impl<W> SortWriter<W> {
    fn sort_batch(&self, batch: RecordBatch) -> Result<RecordBatch> {
        let key_sources = self.projector.project_column(batch.columns())?;

        let mut sort_columns = Vec::with_capacity(key_sources.len());
        for ((source, transform), options) in
            key_sources.iter().zip(&self.transforms).zip(&self.options)
        {
            sort_columns.push(SortColumn {
                values: transform.transform(source.clone())?,
                options: Some(*options),
            });
        }

        let indices = lexsort_to_indices(&sort_columns, None).map_err(arrow_err)?;
        let sorted = batch
            .columns()
            .iter()
            .map(|column| take(column.as_ref(), &indices, None).map_err(arrow_err))
            .collect::<Result<Vec<_>>>()?;

        RecordBatch::try_new(batch.schema(), sorted).map_err(arrow_err)
    }
}

#[async_trait]
impl<W: IcebergWriter> IcebergWriter for SortWriter<W> {
    async fn write(&mut self, input: RecordBatch) -> Result<()> {
        self.buffer.push(input);
        Ok(())
    }

    async fn close(&mut self) -> Result<Vec<DataFile>> {
        let batches = mem::take(&mut self.buffer);
        if !batches.is_empty() {
            let combined = concat_batches(&batches[0].schema(), &batches).map_err(|err| {
                Error::new(
                    ErrorKind::DataInvalid,
                    "Failed to concatenate record batches before sorting; \
                     do all written batches share the same schema?",
                )
                .with_source(err)
            })?;
            // Guard on row count, not batch count: a caller that only writes empty batches would
            // otherwise emit a spurious zero-row data file.
            if combined.num_rows() > 0 {
                let to_write = if self.transforms.is_empty() {
                    combined
                } else {
                    self.sort_batch(combined)?
                };
                self.inner.write(to_write).await?;
            }
        }

        self.inner.close().await
    }
}

fn arrow_err(err: arrow_schema::ArrowError) -> Error {
    Error::new(
        ErrorKind::Unexpected,
        "Arrow operation failed while sorting",
    )
    .with_source(err)
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow_array::{ArrayRef, Int32Array, StringArray};
    use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
    use parquet::file::properties::WriterProperties;
    use tempfile::TempDir;

    use super::*;
    use crate::arrow::schema_to_arrow_schema;
    use crate::io::FileIO;
    use crate::spec::{
        DataFileFormat, NestedField, PrimitiveType, Schema, SortField, SortOrder, Type,
    };
    use crate::writer::base_writer::data_file_writer::DataFileWriterBuilder;
    use crate::writer::file_writer::ParquetWriterBuilder;
    use crate::writer::file_writer::location_generator::{
        DefaultFileNameGenerator, DefaultLocationGenerator,
    };
    use crate::writer::file_writer::rolling_writer::RollingFileWriterBuilder;

    // Schema with two required int columns: "id" (field 1) and "grp" (field 2).
    fn test_schema() -> SchemaRef {
        Arc::new(
            Schema::builder()
                .with_schema_id(1)
                .with_fields(vec![
                    NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int)).into(),
                    NestedField::required(2, "grp", Type::Primitive(PrimitiveType::Int)).into(),
                ])
                .build()
                .unwrap(),
        )
    }

    fn sort_field(
        source_id: i32,
        transform: Transform,
        direction: SortDirection,
        null_order: NullOrder,
    ) -> SortField {
        SortField::builder()
            .source_id(source_id)
            .transform(transform)
            .direction(direction)
            .null_order(null_order)
            .build()
    }

    fn batch(schema: &SchemaRef, id: Vec<i32>, grp: Vec<i32>) -> RecordBatch {
        let arrow_schema = Arc::new(schema_to_arrow_schema(schema).unwrap());
        RecordBatch::try_new(arrow_schema, vec![
            Arc::new(Int32Array::from(id)),
            Arc::new(Int32Array::from(grp)),
        ])
        .unwrap()
    }

    fn column(batch: &RecordBatch, idx: usize) -> Vec<i32> {
        batch
            .column(idx)
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap()
            .values()
            .to_vec()
    }

    // Builds the real writer stack (Parquet -> Rolling -> DataFile) that a SortWriter wraps.
    fn inner_builder(schema: SchemaRef, temp_dir: &TempDir) -> impl IcebergWriterBuilder {
        let location_gen = DefaultLocationGenerator::with_data_location(
            temp_dir.path().to_str().unwrap().to_string(),
        );
        let file_name_gen =
            DefaultFileNameGenerator::new("test".to_string(), None, DataFileFormat::Parquet);
        let parquet_builder =
            ParquetWriterBuilder::new(WriterProperties::builder().build(), schema);
        let rolling_builder = RollingFileWriterBuilder::new_with_default_file_size(
            parquet_builder,
            FileIO::new_with_fs(),
            location_gen,
            file_name_gen,
        );
        DataFileWriterBuilder::new(rolling_builder)
    }

    // Writes `batches` through a SortWriter and reads the resulting data files back into a single
    // record batch, or None if nothing was written.
    async fn write_and_read_back(
        schema: SchemaRef,
        sort_order: SortOrderRef,
        batches: Vec<RecordBatch>,
    ) -> Option<RecordBatch> {
        let temp_dir = TempDir::new().unwrap();
        let sort_builder =
            SortWriterBuilder::new(inner_builder(schema.clone(), &temp_dir), sort_order, schema)
                .unwrap();

        let mut writer = sort_builder.build(None).await.unwrap();
        for b in batches {
            writer.write(b).await.unwrap();
        }
        let data_files = writer.close().await.unwrap();

        if data_files.is_empty() {
            return None;
        }

        let file_io = FileIO::new_with_fs();
        let mut read_batches = Vec::new();
        for file in &data_files {
            let content = file_io
                .new_input(file.file_path.clone())
                .unwrap()
                .read()
                .await
                .unwrap();
            let reader = ParquetRecordBatchReaderBuilder::try_new(content)
                .unwrap()
                .build()
                .unwrap();
            for rb in reader {
                read_batches.push(rb.unwrap());
            }
        }
        Some(concat_batches(&read_batches[0].schema(), &read_batches).unwrap())
    }

    #[tokio::test]
    async fn test_sort_single_field_ascending() {
        let schema = test_schema();
        let sort_order = Arc::new(
            SortOrder::builder()
                .with_sort_field(sort_field(
                    1,
                    Transform::Identity,
                    SortDirection::Ascending,
                    NullOrder::First,
                ))
                .build(schema.as_ref())
                .unwrap(),
        );

        let out = write_and_read_back(schema.clone(), sort_order, vec![
            batch(&schema, vec![3, 1], vec![0, 0]),
            batch(&schema, vec![2, 5, 4], vec![0, 0, 0]),
        ])
        .await
        .unwrap();

        assert_eq!(column(&out, 0), vec![1, 2, 3, 4, 5]);
    }

    #[tokio::test]
    async fn test_sort_multi_field_mixed_direction() {
        let schema = test_schema();
        // grp ascending, then id descending within each grp.
        let sort_order = Arc::new(
            SortOrder::builder()
                .with_sort_field(sort_field(
                    2,
                    Transform::Identity,
                    SortDirection::Ascending,
                    NullOrder::First,
                ))
                .with_sort_field(sort_field(
                    1,
                    Transform::Identity,
                    SortDirection::Descending,
                    NullOrder::Last,
                ))
                .build(schema.as_ref())
                .unwrap(),
        );

        let out = write_and_read_back(schema.clone(), sort_order, vec![
            batch(&schema, vec![10, 5], vec![1, 0]),
            batch(&schema, vec![20, 7], vec![1, 0]),
        ])
        .await
        .unwrap();

        assert_eq!(column(&out, 1), vec![0, 0, 1, 1]); // grp
        assert_eq!(column(&out, 0), vec![7, 5, 20, 10]); // id, descending within grp
    }

    #[tokio::test]
    async fn test_sort_by_transform() {
        let schema = test_schema();
        // truncate(id, 10) ascending, then raw id descending as a tiebreak. The truncate grouping
        // makes the result differ from any ordering on the raw id alone, so this proves the
        // transform is applied to compute the sort key.
        let sort_order = Arc::new(
            SortOrder::builder()
                .with_sort_field(sort_field(
                    1,
                    Transform::Truncate(10),
                    SortDirection::Ascending,
                    NullOrder::First,
                ))
                .with_sort_field(sort_field(
                    1,
                    Transform::Identity,
                    SortDirection::Descending,
                    NullOrder::Last,
                ))
                .build(schema.as_ref())
                .unwrap(),
        );

        let out = write_and_read_back(schema.clone(), sort_order, vec![batch(
            &schema,
            vec![15, 3, 22, 11, 8, 25],
            vec![0, 0, 0, 0, 0, 0],
        )])
        .await
        .unwrap();

        // truncation buckets in ascending order (0, 10, 20), id descending inside each bucket.
        assert_eq!(column(&out, 0), vec![8, 3, 15, 11, 25, 22]);
    }

    #[tokio::test]
    async fn test_unsorted_order_is_pass_through() {
        let schema = test_schema();
        let sort_order = Arc::new(SortOrder::unsorted_order());

        let out = write_and_read_back(schema.clone(), sort_order, vec![batch(
            &schema,
            vec![3, 1, 2],
            vec![0, 0, 0],
        )])
        .await
        .unwrap();

        assert_eq!(column(&out, 0), vec![3, 1, 2]);
    }

    #[tokio::test]
    async fn test_close_without_writes_produces_no_files() {
        let schema = test_schema();
        let sort_order = Arc::new(
            SortOrder::builder()
                .with_sort_field(sort_field(
                    1,
                    Transform::Identity,
                    SortDirection::Ascending,
                    NullOrder::First,
                ))
                .build(schema.as_ref())
                .unwrap(),
        );

        assert!(
            write_and_read_back(schema, sort_order, vec![])
                .await
                .is_none()
        );
    }

    #[tokio::test]
    async fn test_sort_null_ordering() {
        // "id" is required and "val" is nullable; sort by "val" and read the null row's position
        // off the id column (which is never null).
        let schema = Arc::new(
            Schema::builder()
                .with_schema_id(1)
                .with_fields(vec![
                    NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int)).into(),
                    NestedField::optional(2, "val", Type::Primitive(PrimitiveType::Int)).into(),
                ])
                .build()
                .unwrap(),
        );
        let arrow_schema = Arc::new(schema_to_arrow_schema(&schema).unwrap());
        let make_batch = || {
            RecordBatch::try_new(arrow_schema.clone(), vec![
                Arc::new(Int32Array::from(vec![10, 20, 30])),
                Arc::new(Int32Array::from(vec![Some(2), None, Some(1)])),
            ])
            .unwrap()
        };
        let order = |null_order| {
            Arc::new(
                SortOrder::builder()
                    .with_sort_field(sort_field(
                        2,
                        Transform::Identity,
                        SortDirection::Ascending,
                        null_order,
                    ))
                    .build(schema.as_ref())
                    .unwrap(),
            )
        };

        // Ascending, nulls last: val [1, 2, null] -> id [30, 10, 20].
        let out = write_and_read_back(schema.clone(), order(NullOrder::Last), vec![make_batch()])
            .await
            .unwrap();
        assert_eq!(column(&out, 0), vec![30, 10, 20]);

        // Ascending, nulls first: val [null, 1, 2] -> id [20, 30, 10].
        let out = write_and_read_back(schema.clone(), order(NullOrder::First), vec![make_batch()])
            .await
            .unwrap();
        assert_eq!(column(&out, 0), vec![20, 30, 10]);
    }

    #[tokio::test]
    async fn test_sort_by_bucket_transform() {
        let schema = test_schema();
        let sort_order = Arc::new(
            SortOrder::builder()
                .with_sort_field(sort_field(
                    1,
                    Transform::Bucket(4),
                    SortDirection::Ascending,
                    NullOrder::First,
                ))
                .build(schema.as_ref())
                .unwrap(),
        );

        let ids = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let out = write_and_read_back(schema.clone(), sort_order, vec![batch(
            &schema,
            ids.clone(),
            vec![0; ids.len()],
        )])
        .await
        .unwrap();

        // The original rows are written (bucket keys are not persisted), but they must be ordered
        // by bucket(id, 4). Recompute each output id's bucket and assert it is non-decreasing.
        let out_ids = column(&out, 0);
        assert_eq!(out_ids.len(), ids.len());
        let bucket = create_transform_function(&Transform::Bucket(4)).unwrap();
        let buckets = bucket
            .transform(Arc::new(Int32Array::from(out_ids)) as ArrayRef)
            .unwrap();
        let buckets = buckets
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap()
            .values();
        assert!(
            buckets.windows(2).all(|w| w[0] <= w[1]),
            "output not ordered by bucket: {buckets:?}"
        );
    }

    #[tokio::test]
    async fn test_sort_string_column() {
        let schema = Arc::new(
            Schema::builder()
                .with_schema_id(1)
                .with_fields(vec![
                    NestedField::required(1, "id", Type::Primitive(PrimitiveType::Int)).into(),
                    NestedField::required(2, "name", Type::Primitive(PrimitiveType::String)).into(),
                ])
                .build()
                .unwrap(),
        );
        let arrow_schema = Arc::new(schema_to_arrow_schema(&schema).unwrap());
        let data = RecordBatch::try_new(arrow_schema, vec![
            Arc::new(Int32Array::from(vec![1, 2, 3])),
            Arc::new(StringArray::from(vec!["charlie", "alice", "bob"])),
        ])
        .unwrap();
        let sort_order = Arc::new(
            SortOrder::builder()
                .with_sort_field(sort_field(
                    2,
                    Transform::Identity,
                    SortDirection::Ascending,
                    NullOrder::First,
                ))
                .build(schema.as_ref())
                .unwrap(),
        );

        let out = write_and_read_back(schema.clone(), sort_order, vec![data])
            .await
            .unwrap();

        // name ascending -> id order [2 (alice), 3 (bob), 1 (charlie)].
        assert_eq!(column(&out, 0), vec![2, 3, 1]);
    }

    #[test]
    fn test_builder_rejects_unknown_source_column() {
        let schema = test_schema();
        let temp_dir = TempDir::new().unwrap();
        let bad_order = Arc::new(
            SortOrder::builder()
                .with_sort_field(sort_field(
                    999,
                    Transform::Identity,
                    SortDirection::Ascending,
                    NullOrder::First,
                ))
                .build_unbound()
                .unwrap(),
        );

        let result =
            SortWriterBuilder::new(inner_builder(schema.clone(), &temp_dir), bad_order, schema);
        assert!(result.is_err());
    }
}
