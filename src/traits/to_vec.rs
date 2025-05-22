use std::ops::Range;
use crate::axes_traits::AxisType;
use crate::{Axis, RawDataType};

pub(crate) trait ToVec<T> {
    fn to_vec(self) -> Vec<T>;
}

impl<T: RawDataType> ToVec<T> for T {
    fn to_vec(self) -> Vec<T> {
        vec![self]
    }
}

impl ToVec<isize> for Axis {
    fn to_vec(self) -> Vec<isize> {
        vec![self.isize()]
    }
}

impl<T> ToVec<T> for Vec<T> {
    fn to_vec(self) -> Vec<T> { self }
}

impl<T: Clone> ToVec<T> for &[T] {
    fn to_vec(self) -> Vec<T> { Vec::from(self) }
}

impl<T, const N: usize> ToVec<T> for [T; N] {
    fn to_vec(self) -> Vec<T> {
        Vec::from(self)
    }
}

impl ToVec<usize> for Range<usize> {
    fn to_vec(self) -> Vec<usize> {
        self.collect()
    }
}
