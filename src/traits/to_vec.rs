use crate::axes_traits::AxisType;
use crate::Axis;

pub(crate) trait ToVec<T> {
    fn to_vec(self) -> Vec<T>;
}

impl ToVec<usize> for usize {
    fn to_vec(self) -> Vec<usize> {
        vec![self]
    }
}

impl ToVec<usize> for Axis {
    fn to_vec(self) -> Vec<usize> {
        vec![self.usize()]
    }
}

impl<T> ToVec<T> for Vec<T> {
    fn to_vec(self) -> Vec<T> {
        self
    }
}

impl<T, const N: usize> ToVec<T> for [T; N] {
    fn to_vec(self) -> Vec<T> {
        Vec::from(self)
    }
}
