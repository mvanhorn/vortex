// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_buffer::buffer;
use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::IntoArray;
use crate::RecursiveCanonical;
use crate::VortexSessionExecute;
use crate::array_session;
use crate::arrays::ConstantArray;
use crate::arrays::PrimitiveArray;
use crate::assert_arrays_eq;
use crate::builtins::ArrayBuiltins;
use crate::scalar::Scalar;
use crate::scalar_fn::fns::operators::Operator;
use crate::validity::Validity;

fn sub_scalar(array: &ArrayRef, scalar: impl Into<Scalar>) -> VortexResult<ArrayRef> {
    array
        .binary(
            ConstantArray::new(scalar, array.len()).into_array(),
            Operator::Sub,
        )
        .and_then(|a| a.execute::<RecursiveCanonical>(&mut array_session().create_execution_ctx()))
        .map(|a| a.0.into_array())
}

#[test]
fn test_scalar_subtract_unsigned() {
    let mut ctx = array_session().create_execution_ctx();
    let values = buffer![1u16, 2, 3].into_array();
    let result = sub_scalar(&values, 1u16).unwrap();
    assert_arrays_eq!(result, PrimitiveArray::from_iter([0u16, 1, 2]), &mut ctx);
}

#[test]
fn test_scalar_subtract_signed() {
    let mut ctx = array_session().create_execution_ctx();
    let values = buffer![1i64, 2, 3].into_array();
    let result = sub_scalar(&values, -1i64).unwrap();
    assert_arrays_eq!(result, PrimitiveArray::from_iter([2i64, 3, 4]), &mut ctx);
}

#[test]
fn test_scalar_subtract_nullable() {
    let mut ctx = array_session().create_execution_ctx();
    let values = PrimitiveArray::from_option_iter([Some(1u16), Some(2), None, Some(3)]);
    let result = sub_scalar(&values.into_array(), Some(1u16)).unwrap();
    assert_arrays_eq!(
        result,
        PrimitiveArray::from_option_iter([Some(0u16), Some(1), None, Some(2)]),
        &mut ctx
    );
}

#[test]
fn test_scalar_subtract_float() {
    let mut ctx = array_session().create_execution_ctx();
    let values = buffer![1.0f64, 2.0, 3.0].into_array();
    let result = sub_scalar(&values, -1f64).unwrap();
    assert_arrays_eq!(
        result,
        PrimitiveArray::from_iter([2.0f64, 3.0, 4.0]),
        &mut ctx
    );
}

#[test]
fn test_scalar_subtract_float_underflow_is_ok() {
    let values = buffer![f32::MIN, 2.0, 3.0].into_array();
    let _results = sub_scalar(&values, 1.0f32).unwrap();
    let _results = sub_scalar(&values, f32::MAX).unwrap();
}

#[test]
fn test_float_divide_by_zero_is_ok() {
    let mut ctx = array_session().create_execution_ctx();
    let values = buffer![1.0f64, -1.0].into_array();
    let result = values
        .binary(
            ConstantArray::new(0.0f64, values.len()).into_array(),
            Operator::Div,
        )
        .and_then(|a| a.execute::<PrimitiveArray>(&mut array_session().create_execution_ctx()))
        .unwrap();

    assert_arrays_eq!(
        result,
        PrimitiveArray::from_iter([f64::INFINITY, f64::NEG_INFINITY]),
        &mut ctx
    );
}

#[test]
fn test_integer_overflow_errors() {
    let values = buffer![u8::MAX].into_array();
    let result = values
        .binary(
            ConstantArray::new(1u8, values.len()).into_array(),
            Operator::Add,
        )
        .and_then(|a| a.execute::<PrimitiveArray>(&mut array_session().create_execution_ctx()));

    assert!(result.is_err());
}

#[test]
fn test_integer_divide_by_zero_errors() {
    let values = buffer![1i32].into_array();
    let result = values
        .binary(
            ConstantArray::new(0i32, values.len()).into_array(),
            Operator::Div,
        )
        .and_then(|a| a.execute::<PrimitiveArray>(&mut array_session().create_execution_ctx()));

    assert!(result.is_err());
}

#[test]
fn test_integer_divide_overflow_errors() {
    let values = buffer![i64::MIN].into_array();
    let result = values
        .binary(
            ConstantArray::new(-1i64, values.len()).into_array(),
            Operator::Div,
        )
        .and_then(|a| a.execute::<PrimitiveArray>(&mut array_session().create_execution_ctx()));

    assert!(result.is_err());
}

#[test]
fn test_integer_divide_errors_ignore_null_lanes() {
    let mut ctx = array_session().create_execution_ctx();
    let lhs =
        PrimitiveArray::new(buffer![10i32, 10], Validity::from_iter([false, true])).into_array();
    let rhs = buffer![0i32, 2].into_array();
    let result = lhs
        .binary(rhs, Operator::Div)
        .and_then(|a| a.execute::<RecursiveCanonical>(&mut array_session().create_execution_ctx()))
        .map(|a| a.0.into_array())
        .unwrap();

    assert_arrays_eq!(
        result,
        PrimitiveArray::from_option_iter([None, Some(5i32)]),
        &mut ctx
    );
}

#[test]
fn test_integer_errors_ignore_null_lanes() {
    let mut ctx = array_session().create_execution_ctx();
    let values =
        PrimitiveArray::new(buffer![u8::MAX, 1], Validity::from_iter([false, true])).into_array();
    let result = values
        .binary(
            ConstantArray::new(1u8, values.len()).into_array(),
            Operator::Add,
        )
        .and_then(|a| a.execute::<RecursiveCanonical>(&mut array_session().create_execution_ctx()))
        .map(|a| a.0.into_array())
        .unwrap();

    assert_arrays_eq!(
        result,
        PrimitiveArray::from_option_iter([None, Some(2u8)]),
        &mut ctx
    );
}

#[test]
fn test_integer_array_array_errors_on_valid_lanes() {
    let lhs = PrimitiveArray::new(
        buffer![u8::MAX, 1, u8::MAX],
        Validity::from_iter([false, true, true]),
    )
    .into_array();
    let rhs = buffer![1u8, 1, 1].into_array();
    let result = lhs
        .binary(rhs, Operator::Add)
        .and_then(|a| a.execute::<PrimitiveArray>(&mut array_session().create_execution_ctx()));

    assert!(result.is_err());
}

#[test]
fn test_present_nullable_constant_preserves_nullable_output() {
    let mut ctx = array_session().create_execution_ctx();
    let values = buffer![1u8, 2].into_array();
    let result = values
        .binary(
            ConstantArray::new(Some(1u8), values.len()).into_array(),
            Operator::Add,
        )
        .and_then(|a| a.execute::<PrimitiveArray>(&mut array_session().create_execution_ctx()))
        .unwrap();

    assert_arrays_eq!(
        result,
        PrimitiveArray::from_option_iter([Some(2u8), Some(3)]),
        &mut ctx
    );
}
