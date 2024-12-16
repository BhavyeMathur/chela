use crate::data_buffer::DataBuffer;
use crate::TensorBase;
use std::ops::Index;

impl<T, const D: usize> Index<[usize; D]> for TensorBase<T>
where
    T: DataBuffer,
{
    type Output = <T as Index<usize>>::Output;

    fn index(&self, index: [usize; D]) -> &Self::Output {
        assert_eq!(
            D, self.ndims,
            "[] index must equal number of tensor dimensions!"
        );

        let i: usize = index.iter().zip(self.stride.iter())
            .map(|(idx, stride)| idx * stride)
            .sum();

        &self.data[i]
    }
}

impl<T: DataBuffer> Index<usize> for TensorBase<T> {
    type Output = <T as Index<usize>>::Output;

    fn index(&self, index: usize) -> &Self::Output {
        &self[[index]]
    }
}
