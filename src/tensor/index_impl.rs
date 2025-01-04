use crate::dtype::RawDataType;
use crate::Tensor;
use std::ops::Index;

impl<T: RawDataType, const D: usize> Index<[usize; D]> for Tensor<T> {
    type Output = T;

    fn index(&self, index: [usize; D]) -> &Self::Output {
        assert_eq!(D, self.ndims(), "[] index must equal number of tensor dimensions!");

        let i: usize = index.iter().zip(self.stride.iter())
            .map(|(idx, stride)| idx * stride)
            .sum();

        assert!(i < self.len, "[] index out of bounds!");
        unsafe { self.ptr.add(i).as_mut() }
    }
}

impl<T: RawDataType> Index<usize> for Tensor<T> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        &self[[index]]
    }
}
