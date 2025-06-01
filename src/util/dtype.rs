use crate::common::binary_ops::BinaryOps;
use crate::linalg::matrix_ops::MatrixOps;
use crate::ops::dot_product::DotProduct;
use crate::ops::reduce_product::ReduceProduct;
use crate::ops::reduce_sum::ReduceSum;
use crate::sum_of_products::SumOfProductsType;
use num::traits::MulAdd;
use num::{Bounded, Float, NumCast, ToPrimitive};
use rand::distributions::uniform::SampleUniform;
use std::fmt::{Debug, Display};
use std::iter::{Product, Sum};
use std::ops::{Div, Neg, Sub, SubAssign};

pub trait RawDataType: Clone + Copy + PartialEq + Display + Default + Debug + Send + Sync + BinaryOps<Self> + 'static {}

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

pub trait NumericDataType: RawDataType + ToPrimitive + PartialOrd + Bounded + NumCast
+ Sum + Product + SubAssign + From<bool> + ReduceSum + ReduceProduct + DotProduct
+ Sub<Output=Self> + Div<Output=Self> + MulAdd<Output=Self>
{
    type AsFloatType: FloatDataType;

    fn to_float(&self) -> Self::AsFloatType {
        self.to_f32().unwrap().into()
    }

    fn abs(&self) -> Self;

    fn ceil(&self) -> Self {
        *self
    }

    fn floor(&self) -> Self {
        *self
    }
}

impl NumericDataType for u8 {
    type AsFloatType = f32;

    fn abs(&self) -> Self {
        *self
    }
}

impl NumericDataType for u16 {
    type AsFloatType = f32;

    fn abs(&self) -> Self {
        *self
    }
}

impl NumericDataType for u32 {
    type AsFloatType = f32;

    fn abs(&self) -> Self {
        *self
    }
}

impl NumericDataType for u64 {
    type AsFloatType = f64;

    fn abs(&self) -> Self {
        *self
    }
}

impl NumericDataType for u128 {
    type AsFloatType = f64;

    fn abs(&self) -> Self {
        *self
    }
}

impl NumericDataType for usize {
    type AsFloatType = f64;

    fn abs(&self) -> Self {
        *self
    }
}

impl NumericDataType for i8 {
    type AsFloatType = f32;

    fn abs(&self) -> Self {
        num::Signed::abs(self)
    }
}

impl NumericDataType for i16 {
    type AsFloatType = f32;

    fn abs(&self) -> Self {
        num::Signed::abs(self)
    }
}

impl NumericDataType for i32 {
    type AsFloatType = f32;

    fn abs(&self) -> Self {
        num::Signed::abs(self)
    }
}

impl NumericDataType for i64 {
    type AsFloatType = f64;

    fn abs(&self) -> Self {
        num::Signed::abs(self)
    }
}

impl NumericDataType for i128 {
    type AsFloatType = f64;

    fn abs(&self) -> Self {
        num::Signed::abs(self)
    }
}

impl NumericDataType for isize {
    type AsFloatType = f64;

    fn abs(&self) -> Self {
        num::Signed::abs(self)
    }
}

impl NumericDataType for f32 {
    type AsFloatType = f32;

    fn abs(&self) -> Self {
        num::Signed::abs(self)
    }

    fn ceil(&self) -> Self {
        num::Float::ceil(*self)
    }

    fn floor(&self) -> Self {
        num::Float::floor(*self)
    }
}

impl NumericDataType for f64 {
    type AsFloatType = f64;

    fn abs(&self) -> Self {
        num::Signed::abs(self)
    }

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
+ SumOfProductsType + MatrixOps {}

impl FloatDataType for f32 {}
impl FloatDataType for f64 {}


pub trait TensorDataType: FloatDataType {}
impl<T: FloatDataType> TensorDataType for T {}
