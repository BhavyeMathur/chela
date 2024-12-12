use crate::tensor::data_owned::DataOwned;
use crate::tensor::dtype::RawDataType;
use crate::tensor::Tensor;
use std::ops::Index;

impl<T> Index<usize> for DataOwned<T>
where
    T: RawDataType,
{
    type Output = T;

    fn index(&self, index: usize) -> &T {
        assert!(0 <= index && index < self.len, "Index '{index}' out of bounds");
        unsafe { &*self.ptr.as_ptr().offset(index as isize) }
    }
}

impl<T, const D: usize> Index<[usize; D]> for Tensor<T>
where
    T: RawDataType,
{
    type Output = T;

    fn index(&self, index: [usize; D]) -> &T {
        assert_eq!(
            D, self.ndims,
            "[] index must equal number of tensor dimensions!"
        );

        let mut i = 0;
        for dim in 0..D {
            i += index[dim] * self.stride[dim];
        }

        &self.data[i]
    }
}

impl<T: RawDataType> Index<usize> for Tensor<T> {
    type Output = T;

    fn index(&self, index: usize) -> &T {
        &self[[index]]
    }
}
