use num::{Bounded, ToPrimitive};
use std::fmt::{Debug, Display};
use std::iter::{Product, Sum};
use std::ops::{Add, AddAssign, Div, Mul, MulAssign};

pub trait RawDataType: Clone + Copy + PartialEq + Display + Default + Debug + Send + Sync {}

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

impl RawDataType for f32 {}
impl RawDataType for f64 {}

impl RawDataType for bool {}

pub trait NumericDataType:
RawDataType + Sum + Product + Add<Output=Self> + Mul<Output=Self> + Div<Output=Self> + AddAssign + MulAssign
+ ToPrimitive + PartialOrd + Bounded + num::Zero + num::One
{
    type AsFloatType: NumericDataType + From<f32>;

    fn to_float(&self) -> Self::AsFloatType {
        self.to_f32().unwrap().into()
    }

    fn abs(&self) -> Self;
}

impl NumericDataType for u8 {
    type AsFloatType = f32;

    fn abs(&self) -> Self {
        self.clone()
    }
}

impl NumericDataType for u16 {
    type AsFloatType = f32;

    fn abs(&self) -> Self {
        self.clone()
    }
}

impl NumericDataType for u32 {
    type AsFloatType = f32;

    fn abs(&self) -> Self {
        self.clone()
    }
}

impl NumericDataType for u64 {
    type AsFloatType = f32;

    fn abs(&self) -> Self {
        self.clone()
    }
}

impl NumericDataType for u128 {
    type AsFloatType = f32;

    fn abs(&self) -> Self {
        self.clone()
    }
}

impl NumericDataType for usize {
    type AsFloatType = f32;

    fn abs(&self) -> Self {
        self.clone()
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
    type AsFloatType = f32;

    fn abs(&self) -> Self {
        num::Signed::abs(self)
    }
}

impl NumericDataType for i128 {
    type AsFloatType = f32;

    fn abs(&self) -> Self {
        num::Signed::abs(self)
    }
}

impl NumericDataType for f32 {
    type AsFloatType = f32;

    fn abs(&self) -> Self {
        num::Signed::abs(self)
    }
}

impl NumericDataType for f64 {
    type AsFloatType = f64;

    fn abs(&self) -> Self {
        num::Signed::abs(self)
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

pub trait FloatDataType: NumericDataType {}

impl FloatDataType for f32 {}
impl FloatDataType for f64 {}
