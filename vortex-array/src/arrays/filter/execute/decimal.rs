// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use vortex_error::VortexExpect;
use vortex_mask::MaskValues;

use crate::arrays::DecimalArray;
use crate::arrays::filter::execute::buffer;
use crate::arrays::filter::execute::filter_validity;
use crate::match_each_decimal_value_type;

pub fn filter_decimal(array: &DecimalArray, mask: &Arc<MaskValues>) -> DecimalArray {
    let filtered_validity = filter_validity(
        array
            .validity()
            .vortex_expect("decimal validity should be derivable"),
        mask,
    );

    match_each_decimal_value_type!(array.values_type(), |T| {
        let filtered_buffer = buffer::filter_buffer(array.buffer::<T>(), mask.as_ref());

        // SAFETY: We filter both the validity and the buffer with the same mask, so they must have
        // the same length, and since the buffer came from an existing and valid `DecimalArray` the
        // values must all be be representable by the decimal type.
        unsafe {
            DecimalArray::new_unchecked(filtered_buffer, array.decimal_dtype(), filtered_validity)
        }
    })
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use crate::IntoArray;
    use crate::VortexSessionExecute;
    use crate::array_session;
    use crate::arrays::filter::execute::decimal::DecimalArray;
    use crate::compute::conformance::filter::test_filter_conformance;
    use crate::dtype::DecimalDType;
    use crate::dtype::i256;

    #[rstest]
    #[case(DecimalArray::from_iter(
        [1i8, 2, 3, 4, 5],
        DecimalDType::new(2, 0),
    ))]
    #[case(DecimalArray::from_iter(
        [10i16, 20, 30, 40, 50],
        DecimalDType::new(3, 0),
    ))]
    #[case(DecimalArray::from_iter(
        [100i32, 200, 300, 400, 500],
        DecimalDType::new(5, 0),
    ))]
    #[case(DecimalArray::from_iter(
        [1_000i64, 2_000, 3_000, 4_000, 5_000],
        DecimalDType::new(10, 0),
    ))]
    #[case(DecimalArray::from_iter(
        [10_000i128, 20_000, 30_000, 40_000, 50_000],
        DecimalDType::new(19, 0),
    ))]
    #[case(DecimalArray::from_iter(
        [1i128, 2, 3, 4, 5].map(i256::from_i128),
        DecimalDType::new(39, 0),
    ))]
    fn test_filter_decimal_physical_type_conformance(#[case] array: DecimalArray) {
        test_filter_conformance(
            &array.into_array(),
            &mut array_session().create_execution_ctx(),
        );
    }

    #[test]
    fn test_filter_decimal128_conformance() {
        let decimal_dtype = DecimalDType::new(38, 2);
        let values = vec![
            Some(12345i128),
            Some(67890),
            Some(-12345),
            Some(0),
            Some(99999),
        ];
        let array = DecimalArray::from_option_iter(values, decimal_dtype);
        test_filter_conformance(
            &array.into_array(),
            &mut array_session().create_execution_ctx(),
        );
    }

    #[test]
    fn test_filter_decimal128_with_nulls_conformance() {
        let decimal_dtype = DecimalDType::new(38, 4);
        let values = vec![Some(12345i128), None, Some(-12345), Some(0), None];
        let array = DecimalArray::from_option_iter(values, decimal_dtype);
        test_filter_conformance(
            &array.into_array(),
            &mut array_session().create_execution_ctx(),
        );
    }
}
