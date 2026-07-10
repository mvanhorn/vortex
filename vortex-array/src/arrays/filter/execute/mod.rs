// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Execution logic for [`super::FilterArray`].
//!
//! The main entrypoint is [`execute_filter`] which filters any [`Canonical`] array.

use std::ops::Range;
use std::sync::Arc;

use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_mask::Mask;
use vortex_mask::MaskValues;

use crate::ArrayRef;
use crate::Canonical;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::array::ArrayView;
use crate::arrays::ConstantArray;
use crate::arrays::ExtensionArray;
use crate::arrays::Filter;
use crate::arrays::NullArray;
use crate::arrays::VariantArray;
use crate::arrays::extension::ExtensionArrayExt;
use crate::arrays::filter::FilterArrayExt;
use crate::arrays::variant::VariantArrayExt;
use crate::scalar::Scalar;
use crate::validity::Validity;

mod bitbuffer;
mod bool;
mod buffer;
pub(crate) mod byte_compress;
mod decimal;
mod fixed_size_list;
mod listview;
mod primitive;
mod slice;
mod struct_;
pub mod take;
mod varbinview;

/// A helper function that lazily filters a [`Validity`] with selection mask values.
fn filter_validity(validity: Validity, mask: &Arc<MaskValues>) -> Validity {
    validity
        .filter(&Mask::Values(Arc::clone(mask)))
        .vortex_expect("Somehow unable to wrap filter around a validity array")
}

pub(super) fn contiguous_filter_range(mask: &Mask) -> Option<Range<usize>> {
    let start = mask.first()?;
    let end = mask.last()?.checked_add(1)?;
    (end - start == mask.true_count()).then_some(start..end)
}

pub(super) fn prepare_mask_for_reuse(mask: &MaskValues, consumers: usize) {
    buffer::prepare_mask_for_reuse(mask, consumers);
}

/// Check for some fast-path execution conditions before calling [`execute_filter`].
pub(super) fn execute_filter_fast_paths(
    array: ArrayView<'_, Filter>,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Option<ArrayRef>> {
    let true_count = array.mask.true_count();

    // If the mask selects nothing, the output is empty.
    if true_count == 0 {
        return Ok(Some(Canonical::empty(array.dtype()).into_array()));
    }

    // If the mask selects everything, then we can just fully decompress the whole thing.
    if true_count == array.mask.len() {
        return Ok(Some(array.child().clone()));
    }

    // Filtering by one contiguous range is exactly a slice and can remain zero-copy.
    if let Some(range) = contiguous_filter_range(array.filter_mask()) {
        return array.child().slice(range).map(Some);
    }

    // Also check if the array itself is completely null, in which case we only care about the total
    // number of nulls, not the values.
    let child_arr = array.array();
    if child_arr
        .validity()?
        .execute_mask(child_arr.len(), ctx)?
        .true_count()
        == 0
    {
        return Ok(Some(
            ConstantArray::new(Scalar::null(array.dtype().clone()), true_count).into_array(),
        ));
    }

    Ok(None)
}

/// Filter a canonical array by a mask, returning a new canonical array.
pub(super) fn execute_filter(canonical: Canonical, mask: &Arc<MaskValues>) -> Canonical {
    match canonical {
        Canonical::Null(_) => Canonical::Null(NullArray::new(mask.true_count())),
        Canonical::Bool(a) => Canonical::Bool(bool::filter_bool(&a, mask)),
        Canonical::Primitive(a) => Canonical::Primitive(primitive::filter_primitive(&a, mask)),
        Canonical::Decimal(a) => Canonical::Decimal(decimal::filter_decimal(&a, mask)),
        Canonical::VarBinView(a) => Canonical::VarBinView(varbinview::filter_varbinview(&a, mask)),
        Canonical::List(a) => Canonical::List(listview::filter_listview(&a, mask)),
        Canonical::FixedSizeList(a) => {
            Canonical::FixedSizeList(fixed_size_list::filter_fixed_size_list(&a, mask))
        }
        Canonical::Struct(a) => Canonical::Struct(struct_::filter_struct(&a, mask)),
        Canonical::Extension(a) => {
            let filtered_storage = a
                .storage_array()
                .filter(Mask::Values(Arc::clone(mask)))
                .vortex_expect("ExtensionArray storage type somehow could not be filtered");
            Canonical::Extension(ExtensionArray::new(a.ext_dtype().clone(), filtered_storage))
        }
        Canonical::Variant(a) => {
            let filter_mask = Mask::Values(Arc::clone(mask));
            let filtered_core_storage = a
                .core_storage()
                .filter(filter_mask.clone())
                .vortex_expect("VariantArray core_storage could not be filtered");
            let filtered_shredded = a.shredded().map(|shredded| {
                shredded
                    .filter(filter_mask)
                    .vortex_expect("VariantArray shredded child could not be filtered")
            });
            Canonical::Variant(
                VariantArray::try_new(filtered_core_storage, filtered_shredded)
                    .vortex_expect("filtered VariantArray children are row-aligned"),
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use vortex_error::VortexResult;

    use super::*;
    use crate::VortexSessionExecute;
    use crate::array_session;
    use crate::arrays::PrimitiveArray;

    #[test]
    fn contiguous_filter_executes_as_zero_copy_slice() -> VortexResult<()> {
        let array = PrimitiveArray::from_iter(0i32..8);
        let original = array.to_buffer::<i32>();
        let filtered = array
            .into_array()
            .filter(Mask::from_slices(8, vec![(2, 6)]))?
            .execute::<PrimitiveArray>(&mut array_session().create_execution_ctx())?;
        let filtered_values = filtered.to_buffer::<i32>();

        assert_eq!(filtered_values.as_slice(), &[2, 3, 4, 5]);
        assert_eq!(filtered_values.as_ptr(), original.as_ptr().wrapping_add(2));
        Ok(())
    }

    #[test]
    fn fragmented_filter_is_not_a_contiguous_range() {
        let mask = Mask::from_indices(8, [1, 2, 5, 6]);
        assert_eq!(contiguous_filter_range(&mask), None);
    }
}
