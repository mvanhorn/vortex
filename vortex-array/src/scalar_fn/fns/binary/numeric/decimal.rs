// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Native execution of the arithmetic operators over decimal arrays.
//!
//! Both operands share a logical [`DecimalDType`] (equal precision and scale). Add and Sub apply
//! directly to the unscaled stored integers and are exact at that shared scale. The result reserves
//! one additional precision digit for a carry, capped at Vortex's maximum decimal precision. Mul
//! and Div require rescaling and are not yet implemented.
//!
//! Lanes execute in a working width chosen so that in-precision inputs cannot spuriously
//! overflow an intermediate value. An operation that overflows the result precision on a valid
//! lane is an error; invalid lanes never error.

use num_traits::CheckedAdd;
use num_traits::CheckedSub;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;
use vortex_mask::AllOr;
use vortex_mask::Mask;

use super::CheckedValues;
use super::check_numeric_errors;
use super::checked_all_lanes;
use super::checked_valid_lanes;
use super::decimal_add_sub_result_dtype;
use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::arrays::Constant;
use crate::arrays::ConstantArray;
use crate::arrays::DecimalArray;
use crate::arrays::decimal::DecimalArrayExt;
use crate::dtype::BigCast;
use crate::dtype::DType;
use crate::dtype::DecimalDType;
use crate::dtype::DecimalType;
use crate::dtype::NativeDecimalType;
use crate::dtype::i256;
use crate::match_each_decimal_value_type;
use crate::scalar::DecimalValue;
use crate::scalar::NumericOperator;
use crate::scalar::Scalar;
use crate::validity::Validity;

/// Execute a numeric operation between two decimal arrays sharing a decimal dtype.
pub(super) fn execute_numeric_decimal(
    lhs: &ArrayRef,
    rhs: &ArrayRef,
    op: NumericOperator,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    let DType::Decimal(decimal_dtype, _) = lhs.dtype() else {
        vortex_bail!("expected a decimal dtype, got {}", lhs.dtype());
    };
    let result_decimal_dtype = decimal_add_sub_result_dtype(*decimal_dtype);
    let result_dtype = DType::Decimal(
        result_decimal_dtype,
        lhs.dtype().nullability() | rhs.dtype().nullability(),
    );

    let lhs = DecimalOperand::try_new(lhs, ctx)?;
    let rhs = DecimalOperand::try_new(rhs, ctx)?;
    let len = lhs.len();
    debug_assert_eq!(len, rhs.len());

    let validity = lhs.validity().and(rhs.validity())?;
    let valid_rows = validity.execute_mask(len, ctx)?;

    match DecimalType::smallest_decimal_value_type(&result_decimal_dtype) {
        DecimalType::I8 => execute_decimal_at_widths::<i8, i8>(
            &lhs,
            &rhs,
            op,
            result_decimal_dtype,
            &result_dtype,
            validity,
            &valid_rows,
        ),
        DecimalType::I16 => execute_decimal_at_widths::<i16, i16>(
            &lhs,
            &rhs,
            op,
            result_decimal_dtype,
            &result_dtype,
            validity,
            &valid_rows,
        ),
        DecimalType::I32 => execute_decimal_at_widths::<i32, i32>(
            &lhs,
            &rhs,
            op,
            result_decimal_dtype,
            &result_dtype,
            validity,
            &valid_rows,
        ),
        DecimalType::I64 => execute_decimal_at_widths::<i64, i64>(
            &lhs,
            &rhs,
            op,
            result_decimal_dtype,
            &result_dtype,
            validity,
            &valid_rows,
        ),
        DecimalType::I128 => execute_decimal_at_widths::<i128, i128>(
            &lhs,
            &rhs,
            op,
            result_decimal_dtype,
            &result_dtype,
            validity,
            &valid_rows,
        ),
        DecimalType::I256 => execute_decimal_at_widths::<i256, i256>(
            &lhs,
            &rhs,
            op,
            result_decimal_dtype,
            &result_dtype,
            validity,
            &valid_rows,
        ),
    }
}

/// A decimal binary-operator operand: a canonical decimal array, a non-null constant, or an
/// all-null constant.
enum DecimalOperand {
    Array {
        values: DecimalArray,
        validity: Validity,
    },
    Constant {
        value: DecimalValue,
        len: usize,
        validity: Validity,
    },
    Null(usize),
}

impl DecimalOperand {
    fn try_new(array: &ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<Self> {
        if let Some(constant) = array.as_opt::<Constant>() {
            return Ok(match constant.scalar().as_decimal().decimal_value() {
                Some(value) => Self::Constant {
                    value,
                    len: array.len(),
                    validity: if constant.scalar().dtype().is_nullable() {
                        Validity::AllValid
                    } else {
                        Validity::NonNullable
                    },
                },
                None => Self::Null(array.len()),
            });
        }

        let values = array.clone().execute::<DecimalArray>(ctx)?;
        let validity = values.validity()?;
        Ok(Self::Array { values, validity })
    }

    fn len(&self) -> usize {
        match self {
            Self::Array { values, .. } => values.len(),
            Self::Constant { len, .. } | Self::Null(len) => *len,
        }
    }

    fn validity(&self) -> Validity {
        match self {
            Self::Array { validity, .. } | Self::Constant { validity, .. } => validity.clone(),
            Self::Null(_) => Validity::AllInvalid,
        }
    }
}

/// Per-execution constants for checked decimal lane operations at working width `W`.
struct DecimalOpPlan<W> {
    /// Inclusive stored-value bounds implied by the result precision.
    prec_min: W,
    prec_max: W,
}

impl<W: NativeDecimalType> DecimalOpPlan<W> {
    fn new(dtype: DecimalDType) -> Self {
        let precision = dtype.precision() as usize;
        Self {
            prec_min: W::MIN_BY_PRECISION[precision],
            prec_max: W::MAX_BY_PRECISION[precision],
        }
    }

    /// Bounds-check a candidate result against the result precision.
    #[inline(always)]
    fn in_precision(&self, value: W) -> Option<W> {
        (self.prec_min <= value && value <= self.prec_max).then_some(value)
    }
}

/// A checked fixed-point decimal operation on unscaled values at working width `W`.
trait CheckedDecimalOp {
    const ERROR: &'static str;

    fn checked<W>(lhs: W, rhs: W, plan: &DecimalOpPlan<W>) -> Option<W>
    where
        W: NativeDecimalType + CheckedAdd + CheckedSub;
}

struct DecimalAdd;

struct DecimalSub;

impl CheckedDecimalOp for DecimalAdd {
    const ERROR: &'static str = "decimal overflow in checked add";

    #[inline(always)]
    fn checked<W>(lhs: W, rhs: W, plan: &DecimalOpPlan<W>) -> Option<W>
    where
        W: NativeDecimalType + CheckedAdd + CheckedSub,
    {
        plan.in_precision(lhs.checked_add(&rhs)?)
    }
}

impl CheckedDecimalOp for DecimalSub {
    const ERROR: &'static str = "decimal overflow in checked sub";

    #[inline(always)]
    fn checked<W>(lhs: W, rhs: W, plan: &DecimalOpPlan<W>) -> Option<W>
    where
        W: NativeDecimalType + CheckedAdd + CheckedSub,
    {
        plan.in_precision(lhs.checked_sub(&rhs)?)
    }
}

fn execute_decimal_at_widths<W, O>(
    lhs: &DecimalOperand,
    rhs: &DecimalOperand,
    op: NumericOperator,
    result_decimal_dtype: DecimalDType,
    result_dtype: &DType,
    validity: Validity,
    valid_rows: &Mask,
) -> VortexResult<ArrayRef>
where
    W: NativeDecimalType + CheckedAdd + CheckedSub,
    O: NativeDecimalType,
    DecimalValue: From<O>,
{
    match op {
        NumericOperator::Add => execute_decimal_typed::<W, O, DecimalAdd>(
            lhs,
            rhs,
            result_decimal_dtype,
            result_dtype,
            validity,
            valid_rows,
        ),
        NumericOperator::Sub => execute_decimal_typed::<W, O, DecimalSub>(
            lhs,
            rhs,
            result_decimal_dtype,
            result_dtype,
            validity,
            valid_rows,
        ),
        NumericOperator::Mul | NumericOperator::Div => vortex_bail!(
            "numeric operator {:?} is not yet supported for decimal arrays",
            op
        ),
    }
}

fn execute_decimal_typed<W, O, Op>(
    lhs: &DecimalOperand,
    rhs: &DecimalOperand,
    result_decimal_dtype: DecimalDType,
    result_dtype: &DType,
    validity: Validity,
    valid_rows: &Mask,
) -> VortexResult<ArrayRef>
where
    W: NativeDecimalType + CheckedAdd + CheckedSub,
    O: NativeDecimalType,
    DecimalValue: From<O>,
    Op: CheckedDecimalOp,
{
    let len = lhs.len();
    let plan = DecimalOpPlan::<W>::new(result_decimal_dtype);

    let checked = match (lhs, rhs) {
        (DecimalOperand::Array { values: lhs, .. }, DecimalOperand::Array { values: rhs, .. }) => {
            checked_decimal_arrays::<W, O, Op>(lhs, rhs, &plan, valid_rows)
        }
        (DecimalOperand::Array { values: lhs, .. }, DecimalOperand::Constant { value, .. }) => {
            let rhs = typed_constant::<W>(value);
            match_each_decimal_value_type!(lhs.values_type(), |L| {
                let lhs = lhs.buffer::<L>();
                checked_decimal_lanes::<W, O, _>(len, valid_rows, |idx| {
                    Op::checked(cast_work_value::<W, L>(lhs[idx]), rhs, &plan)
                })
            })
        }
        (DecimalOperand::Constant { value, .. }, DecimalOperand::Array { values: rhs, .. }) => {
            let lhs = typed_constant::<W>(value);
            match_each_decimal_value_type!(rhs.values_type(), |R| {
                let rhs = rhs.buffer::<R>();
                checked_decimal_lanes::<W, O, _>(len, valid_rows, |idx| {
                    Op::checked(lhs, cast_work_value::<W, R>(rhs[idx]), &plan)
                })
            })
        }
        (
            DecimalOperand::Constant { value: lhs, .. },
            DecimalOperand::Constant { value: rhs, .. },
        ) => {
            let lhs = typed_constant::<W>(lhs);
            let rhs = typed_constant::<W>(rhs);
            let value = Op::checked(lhs, rhs, &plan)
                .ok_or_else(|| vortex_err!(InvalidArgument: "{}", Op::ERROR))?;
            let value = cast_result_value::<W, O>(value);
            return Ok(ConstantArray::new(
                Scalar::decimal(
                    DecimalValue::from(value),
                    result_decimal_dtype,
                    result_dtype.nullability(),
                ),
                len,
            )
            .into_array());
        }
        (DecimalOperand::Null(_), _) | (_, DecimalOperand::Null(_)) => {
            CheckedValues::<O>::zeroed(len)
        }
    };
    check_numeric_errors(checked.failed, Op::ERROR)?;

    Ok(DecimalArray::new(
        checked.values,
        result_decimal_dtype,
        validity.union_nullability(result_dtype.nullability()),
    )
    .into_array())
}

fn checked_decimal_arrays<W, O, Op>(
    lhs: &DecimalArray,
    rhs: &DecimalArray,
    plan: &DecimalOpPlan<W>,
    valid_rows: &Mask,
) -> CheckedValues<O>
where
    W: NativeDecimalType + CheckedAdd + CheckedSub,
    O: NativeDecimalType,
    Op: CheckedDecimalOp,
{
    let len = lhs.len();
    debug_assert_eq!(len, rhs.len());
    match_each_decimal_value_type!(lhs.values_type(), |L| {
        let lhs = lhs.buffer::<L>();
        match_each_decimal_value_type!(rhs.values_type(), |R| {
            let rhs = rhs.buffer::<R>();
            checked_decimal_lanes::<W, O, _>(len, valid_rows, |idx| {
                Op::checked(
                    cast_work_value::<W, L>(lhs[idx]),
                    cast_work_value::<W, R>(rhs[idx]),
                    plan,
                )
            })
        })
    })
}

fn checked_decimal_lanes<W, O, F>(
    len: usize,
    valid_rows: &Mask,
    mut checked_at: F,
) -> CheckedValues<O>
where
    W: NativeDecimalType,
    O: NativeDecimalType,
    F: FnMut(usize) -> Option<W>,
{
    match valid_rows.bit_buffer() {
        AllOr::All => checked_all_lanes(len, |idx| checked_at(idx).map(cast_result_value::<W, O>)),
        AllOr::None => CheckedValues::<O>::zeroed(len),
        AllOr::Some(valid_bits) => checked_valid_lanes(len, valid_bits, |idx| {
            checked_at(idx).map(cast_result_value::<W, O>)
        }),
    }
}

#[inline(always)]
fn cast_work_value<W: NativeDecimalType, T: NativeDecimalType>(value: T) -> W {
    <W as BigCast>::from(value)
        .vortex_expect("valid decimal input must fit the arithmetic working width")
}

#[inline(always)]
fn cast_result_value<W: NativeDecimalType, O: NativeDecimalType>(value: W) -> O {
    <O as BigCast>::from(value)
        .vortex_expect("precision-checked decimal result must fit the output width")
}

fn typed_constant<W: NativeDecimalType>(value: &DecimalValue) -> W {
    value
        .cast::<W>()
        .vortex_expect("the working width must be able to represent the constant")
}
