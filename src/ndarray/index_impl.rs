use crate::dtype::RawDataType;
use crate::{NdArray, StridedMemory};
use std::ops::{Index, IndexMut};

impl<T: RawDataType, const D: usize> Index<[usize; D]> for NdArray<'_, T> {
    type Output = T;

    fn index(&self, index: [usize; D]) -> &Self::Output {
        assert_eq!(D, self.ndims(), "[] index must equal number of tensor dimensions!");

        let i: usize = index.iter().zip(self.stride.iter())
            .map(|(idx, stride)| idx * stride)
            .sum();

        assert!(i < self.len, "[] index out of bounds!");
        unsafe { self.ptr.add(i).as_ref() }
    }
}

impl<T: RawDataType, const D: usize> IndexMut<[usize; D]> for NdArray<'_, T> {
    fn index_mut(&mut self, index: [usize; D]) -> &mut Self::Output {
        assert_eq!(D, self.ndims(), "[] index must equal number of tensor dimensions!");

        let i: usize = index.iter().zip(self.stride.iter())
                            .map(|(idx, stride)| idx * stride)
                            .sum();

        assert!(i < self.len, "[] index out of bounds!");
        unsafe { self.ptr.add(i).as_mut() }
    }
}

impl<T: RawDataType> Index<usize> for NdArray<'_, T> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        &self[[index]]
    }
}

impl<T: RawDataType> IndexMut<usize> for NdArray<'_, T> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self[[index]]
    }
}
