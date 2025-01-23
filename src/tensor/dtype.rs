use num::ToPrimitive;
use std::fmt::Display;
use std::ops::{Div, Range, RangeInclusive};

pub trait RawDataType: Clone + Copy + PartialEq + Display {}

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
    RawDataType + std::iter::Sum + std::iter::Product + Div<Output = Self> + ToPrimitive + PartialOrd
{
    type AsFloatType: NumericDataType + From<f32>;

    fn to_float(&self) -> Self::AsFloatType {
        self.to_f32().unwrap().into()
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
    type AsFloatType = f32;
}
impl NumericDataType for u128 {
    type AsFloatType = f32;
}

impl NumericDataType for usize {
    type AsFloatType = f32;
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
    type AsFloatType = f32;
}
impl NumericDataType for i128 {
    type AsFloatType = f32;
}

impl NumericDataType for f32 {
    type AsFloatType = f32;
}
impl NumericDataType for f64 {
    type AsFloatType = f64;
}