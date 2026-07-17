// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Row-shaped bounding-box computation over native geometry columns.

use geo::BoundingRect;
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_error::VortexResult;

use crate::extension::geometries;

/// Per-row 2-D axis-aligned bounding boxes (AABBs) of a native geometry column, as `(min, max)`
/// pairs of `[x, y]` corners.
///
/// The per-row counterpart of the [`GeometryAabb`](crate::aggregate_fn::GeometryAabb) aggregate,
/// which unions the whole column into one box. A point yields a degenerate box with `min == max`.
/// A null row, or a geometry with no extent (e.g. an empty polygon), yields `None`: either way
/// the row has no box, and matches nothing in a spatial join.
///
/// A non-geometry column is an error. Any `z`/`m` ordinates carry no bounds, this decodes to 2-D `geo_types`.
pub fn aabbs_2d(
    array: &ArrayRef,
    ctx: &mut ExecutionCtx,
) -> VortexResult<impl Iterator<Item = Option<([f64; 2], [f64; 2])>> + use<>> {
    let len = array.len();
    // Drop the null rows before decoding, since a null row has no geometry to decode (`filter`
    // collapses an all-true mask, so an all-valid column passes through unchanged), then weave
    // the boxes back into row order with `None` in every null slot.
    let valid = array.validity()?.execute_mask(len, ctx)?;
    let decoded = geometries(&array.filter(valid.clone())?, ctx)?;
    let mut boxes = decoded.into_iter().map(|geometry| {
        geometry
            .bounding_rect()
            .map(|rect| ([rect.min().x, rect.min().y], [rect.max().x, rect.max().y]))
    });
    Ok((0..len).map(move |row| {
        if valid.value(row) {
            boxes.next().flatten()
        } else {
            None
        }
    }))
}

#[cfg(test)]
mod tests {
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::ConstantArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::scalar::Scalar;
    use vortex_error::VortexResult;

    use super::aabbs_2d;
    use crate::test_harness::linestring_column;
    use crate::test_harness::multilinestring_column;
    use crate::test_harness::multipoint_column;
    use crate::test_harness::multipolygon_column;
    use crate::test_harness::nullable_multipolygon_column;
    use crate::test_harness::nullable_point_column;
    use crate::test_harness::point_column;
    use crate::test_harness::polygon_column;
    use crate::test_harness::rect_column;

    /// A shorthand box from its four corners.
    fn bounds(xmin: f64, ymin: f64, xmax: f64, ymax: f64) -> Option<([f64; 2], [f64; 2])> {
        Some(([xmin, ymin], [xmax, ymax]))
    }

    /// A point's box is degenerate: both corners are the point itself.
    #[test]
    fn point_box_is_degenerate() -> VortexResult<()> {
        let session = vortex_array::array_session();
        let mut ctx = session.create_execution_ctx();

        let points = point_column(vec![1.0, 3.0], vec![2.0, 4.0])?;
        assert_eq!(
            aabbs_2d(&points, &mut ctx)?.collect::<Vec<_>>(),
            vec![bounds(1.0, 2.0, 1.0, 2.0), bounds(3.0, 4.0, 3.0, 4.0)]
        );
        Ok(())
    }

    /// A polygon's box is its extent: the min/max over every ring vertex, one box per row.
    #[test]
    fn polygon_box_is_extent() -> VortexResult<()> {
        let session = vortex_array::array_session();
        let mut ctx = session.create_execution_ctx();

        let polygons = polygon_column(vec![
            vec![vec![(0.0, 0.0), (4.0, 0.0), (2.0, 5.0)]],
            vec![vec![(-3.0, 7.0), (-2.0, 7.0), (-2.0, 9.0), (-3.0, 9.0)]],
        ])?;
        assert_eq!(
            aabbs_2d(&polygons, &mut ctx)?.collect::<Vec<_>>(),
            vec![bounds(0.0, 0.0, 4.0, 5.0), bounds(-3.0, 7.0, -2.0, 9.0)]
        );
        Ok(())
    }

    /// A `Rect` row is its own bounding box.
    #[test]
    fn rect_box_is_itself() -> VortexResult<()> {
        let session = vortex_array::array_session();
        let mut ctx = session.create_execution_ctx();

        let rects = rect_column(vec![(0.0, 0.0, 2.0, 3.0), (-1.0, -1.0, 1.0, 1.0)])?;
        assert_eq!(
            aabbs_2d(&rects, &mut ctx)?.collect::<Vec<_>>(),
            vec![bounds(0.0, 0.0, 2.0, 3.0), bounds(-1.0, -1.0, 1.0, 1.0)]
        );
        Ok(())
    }

    /// Every multi-vertex native type over the same vertex set yields that set's box, so the
    /// whole type family is covered (`Point` has its own degenerate-box test above).
    #[test]
    fn covers_every_native_geometry_type() -> VortexResult<()> {
        let session = vortex_array::array_session();
        let mut ctx = session.create_execution_ctx();

        let vertices = vec![(1.0, 2.0), (-1.0, 5.0), (3.0, 4.0)];
        let columns = vec![
            linestring_column(vec![vertices.clone()])?,
            multipoint_column(vec![vertices.clone()])?,
            polygon_column(vec![vec![vertices.clone()]])?,
            multilinestring_column(vec![vec![vertices.clone()]])?,
            multipolygon_column(vec![vec![vec![vertices]]])?,
        ];
        for column in columns {
            assert_eq!(
                aabbs_2d(&column, &mut ctx)?.collect::<Vec<_>>(),
                vec![bounds(-1.0, 2.0, 3.0, 5.0)],
                "box mismatch for {}",
                column.dtype()
            );
        }
        Ok(())
    }

    /// An empty geometry (here a zero-part multipolygon) has no extent and yields `None`; other
    /// rows keep their boxes.
    #[test]
    fn empty_geometry_has_no_box() -> VortexResult<()> {
        let session = vortex_array::array_session();
        let mut ctx = session.create_execution_ctx();

        let multipolygons = multipolygon_column(vec![
            vec![vec![vec![(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]]],
            vec![],
        ])?;
        assert_eq!(
            aabbs_2d(&multipolygons, &mut ctx)?.collect::<Vec<_>>(),
            vec![bounds(0.0, 0.0, 1.0, 1.0), None]
        );
        Ok(())
    }

    /// A null geometry row yields `None`, just like an empty geometry: no box, so it matches
    /// nothing in a spatial join. Valid rows keep their boxes.
    #[test]
    fn null_row_yields_none() -> VortexResult<()> {
        let session = vortex_array::array_session();
        let mut ctx = session.create_execution_ctx();

        let points = nullable_point_column(vec![Some((1.0, 2.0)), None, Some((3.0, 4.0))])?;
        assert_eq!(
            aabbs_2d(&points, &mut ctx)?.collect::<Vec<_>>(),
            vec![bounds(1.0, 2.0, 1.0, 2.0), None, bounds(3.0, 4.0, 3.0, 4.0)]
        );
        Ok(())
    }

    /// Valid, empty, and null rows stay positionally aligned: the valid row keeps its box, and
    /// the empty and null rows are both `None`.
    #[test]
    fn mixed_valid_empty_null_rows_align() -> VortexResult<()> {
        let session = vortex_array::array_session();
        let mut ctx = session.create_execution_ctx();

        let multipolygons = nullable_multipolygon_column(vec![
            Some(vec![vec![vec![
                (0.0, 0.0),
                (1.0, 0.0),
                (1.0, 1.0),
                (0.0, 1.0),
            ]]]),
            Some(vec![]),
            None,
        ])?;
        assert_eq!(
            aabbs_2d(&multipolygons, &mut ctx)?.collect::<Vec<_>>(),
            vec![bounds(0.0, 0.0, 1.0, 1.0), None, None]
        );
        Ok(())
    }

    /// A constant-null column carries its nulls in the array's logical validity rather than in
    /// materialized storage; every row is `None`.
    #[test]
    fn constant_null_column_is_all_none() -> VortexResult<()> {
        let session = vortex_array::array_session();
        let mut ctx = session.create_execution_ctx();

        let dtype = point_column(vec![0.0], vec![0.0])?.dtype().as_nullable();
        let nulls = ConstantArray::new(Scalar::null(dtype), 2).into_array();
        assert_eq!(
            aabbs_2d(&nulls, &mut ctx)?.collect::<Vec<_>>(),
            vec![None, None]
        );
        Ok(())
    }

    /// A zero-row column yields a zero-row result.
    #[test]
    fn empty_column_yields_nothing() -> VortexResult<()> {
        let session = vortex_array::array_session();
        let mut ctx = session.create_execution_ctx();

        let points = point_column(vec![], vec![])?;
        assert_eq!(aabbs_2d(&points, &mut ctx)?.count(), 0);
        Ok(())
    }

    /// A non-geometry column is an error.
    #[test]
    fn non_geometry_column_is_rejected() -> VortexResult<()> {
        let session = vortex_array::array_session();
        let mut ctx = session.create_execution_ctx();

        let numbers = PrimitiveArray::from_iter([1.0f64, 2.0]).into_array();
        assert!(aabbs_2d(&numbers, &mut ctx).is_err());
        Ok(())
    }
}
