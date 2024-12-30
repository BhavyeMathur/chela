pub(crate) trait ToVec<T> {
    fn to_vec(self) -> Vec<T>;
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
