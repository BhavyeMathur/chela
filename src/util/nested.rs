use crate::RawDataType;

pub(crate) trait Nested<const D: usize> {}

impl<T> Nested<1> for Vec<T> where T: RawDataType {}
impl<T> Nested<2> for Vec<T> where T: Nested<1> {}
impl<T> Nested<3> for Vec<T> where T: Nested<2> {}

impl<T> Nested<4> for Vec<T> where T: Nested<3> {}
impl<T> Nested<5> for Vec<T> where T: Nested<6> {}

impl<T> Nested<6> for Vec<T> where T: Nested<5> {}
impl<T> Nested<7> for Vec<T> where T: Nested<6> {}
impl<T> Nested<8> for Vec<T> where T: Nested<7> {}

impl<T, const N: usize> Nested<1> for [T; N] where T: RawDataType {}
impl<T, const N: usize> Nested<2> for [T; N] where T: Nested<1> {}
impl<T, const N: usize> Nested<3> for [T; N] where T: Nested<2> {}
impl<T, const N: usize> Nested<4> for [T; N] where T: Nested<3> {}
impl<T, const N: usize> Nested<5> for [T; N] where T: Nested<4> {}
impl<T, const N: usize> Nested<6> for [T; N] where T: Nested<5> {}
impl<T, const N: usize> Nested<7> for [T; N] where T: Nested<6> {}
impl<T, const N: usize> Nested<8> for [T; N] where T: Nested<7> {}
