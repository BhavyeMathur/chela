use crate::linalg::matrix_ops::MatrixOps;
use crate::ops::binary_op_add::BinaryOpAdd;
use crate::ops::binary_op_div::BinaryOpDiv;
use crate::ops::binary_op_mul::BinaryOpMul;
use crate::ops::binary_op_sub::BinaryOpSub;
use crate::ops::dot_product::DotProduct;
use crate::ops::fill::Fill;
use crate::ops::reduce_max::ReduceMax;
use crate::ops::reduce_max_magnitude::ReduceMaxMagnitude;
use crate::ops::reduce_min::ReduceMin;
use crate::ops::reduce_min_magnitude::ReduceMinMagnitude;
use crate::ops::reduce_product::ReduceProduct;
use crate::ops::reduce_sum::ReduceSum;
use crate::ops::unary_ops::UnaryOps;
use crate::sum_of_products::SumOfProductsType;
use num::traits::MulAdd;
use num::{Float, NumCast, ToPrimitive};
use rand::distributions::uniform::SampleUniform;
use std::fmt::{Debug, Display};
use std::iter::{Product, Sum};
use std::ops::{Div, Neg, Sub, SubAssign};

pub trait RawDataType: 'static + Default + Copy + Clone + Debug + Display + Sized
+ PartialEq + Fill + Send + Sync {}

impl RawDataType for u8 {}
impl RawDataType for u16 {}
impl RawDataType for u32 {}
impl RawDataType for u64 {}
impl RawDataType for u128 {}
impl RawDataType for usize {}

impl RawDataType for i8 {}
impl RawDataType for i16 {}
impl RawDataType for i32 {}
impl RawDataType for i64 {}
impl RawDataType for i128 {}
impl RawDataType for isize {}

impl RawDataType for f32 {}
impl RawDataType for f64 {}

impl RawDataType for bool {}

pub trait NumericDataType: RawDataType + ToPrimitive + NumCast + From<bool>
+ Sum + Product + SubAssign + Sub<Output=Self> + Div<Output=Self> + MulAdd<Output=Self> + DotProduct
+ ReduceSum + ReduceProduct + ReduceMin + ReduceMax + ReduceMinMagnitude + ReduceMaxMagnitude
+ BinaryOpAdd + BinaryOpSub + BinaryOpMul
{
    type AsFloatType: FloatDataType;

    fn to_float(&self) -> Self::AsFloatType {
        self.to_f32().unwrap().into()
    }

    fn ceil(&self) -> Self {
        *self
    }

    fn floor(&self) -> Self {
        *self
    }
}

impl NumericDataType for u8 {
    type AsFloatType = f32;
}

impl NumericDataType for u16 {
    type AsFloatType = f32;
}

impl NumericDataType for u32 {
    type AsFloatType = f32;
}

impl NumericDataType for u64 {
    type AsFloatType = f64;
}

impl NumericDataType for u128 {
    type AsFloatType = f64;
}

impl NumericDataType for usize {
    type AsFloatType = f64;
}

impl NumericDataType for i8 {
    type AsFloatType = f32;
}

impl NumericDataType for i16 {
    type AsFloatType = f32;
}

impl NumericDataType for i32 {
    type AsFloatType = f32;
}

impl NumericDataType for i64 {
    type AsFloatType = f64;
}

impl NumericDataType for i128 {
    type AsFloatType = f64;
}

impl NumericDataType for isize {
    type AsFloatType = f64;
}

impl NumericDataType for f32 {
    type AsFloatType = f32;

    fn ceil(&self) -> Self {
        num::Float::ceil(*self)
    }

    fn floor(&self) -> Self {
        num::Float::floor(*self)
    }
}

impl NumericDataType for f64 {
    type AsFloatType = f64;

    fn ceil(&self) -> Self {
        num::Float::ceil(*self)
    }

    fn floor(&self) -> Self {
        num::Float::floor(*self)
    }
}

pub trait IntegerDataType: NumericDataType + Ord {}

impl IntegerDataType for u8 {}
impl IntegerDataType for u16 {}
impl IntegerDataType for u32 {}
impl IntegerDataType for u64 {}
impl IntegerDataType for u128 {}
impl IntegerDataType for usize {}

impl IntegerDataType for i8 {}
impl IntegerDataType for i16 {}
impl IntegerDataType for i32 {}
impl IntegerDataType for i64 {}
impl IntegerDataType for i128 {}
impl IntegerDataType for isize {}

pub trait FloatDataType: NumericDataType + Float + From<f32> + SampleUniform + Neg<Output=Self>
+ SumOfProductsType + MatrixOps + BinaryOpDiv + UnaryOps {}

impl FloatDataType for f32 {}
impl FloatDataType for f64 {}


pub trait TensorDataType: FloatDataType {}
impl<T: FloatDataType> TensorDataType for T {}
