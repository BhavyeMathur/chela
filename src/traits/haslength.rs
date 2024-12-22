pub trait HasLength {
    fn len(&self) -> usize;
}

impl<T> HasLength for Vec<T> {
    fn len(&self) -> usize {
        Vec::len(&self)
    }
}
impl<T, const N: usize> HasLength for [T; N] {
    fn len(&self) -> usize {
        N
    }
}
