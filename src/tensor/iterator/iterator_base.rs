use crate::data_buffer::{DataBuffer, DataOwned, DataView};
use crate::dtype::RawDataType;
use crate::{tensor, TensorView};

#[non_exhaustive]
pub struct IteratorBase<'a, T, B>
where
    T: RawDataType,
    B: DataBuffer<DType = T>,
{
    data_buffer: &'a B,
    axis: usize,
    shape: Vec<usize>,
    stride: Vec<usize>,
    indices: usize,
    iter_count: isize,
}

impl<'a, T, B> IteratorBase<'a, T, B>
where
    T: RawDataType,
    B: DataBuffer<DType = T>,
{
    pub(super) fn from(
        data_buffer: &'a B,
        axis: usize,
        shape: Vec<usize>,
        stride: Vec<usize>,
        indices: usize,
    ) -> Self {
        Self {
            data_buffer,
            axis,
            shape,
            stride,
            indices,
            iter_count: 0,
        }
    }
}

impl<'a, T, B> Iterator for IteratorBase<'a, T, B>
where
    T: RawDataType,
    B: DataBuffer<DType = T>,
{
    type Item = TensorView<T>;

    fn next(&mut self) -> Option<Self::Item> {
        match self.iter_count < self.shape[self.axis] as isize {
            false => None,
            true => unsafe {
                let mut ptr_offset = 0isize;
                let mut data_vec: Vec<T> = Vec::new();

                let mut new_shape = self.shape.clone();
                let mut new_stride = self.stride.clone();

                for i in 0..self.axis {
                    new_stride[i] = new_stride[i] / new_shape[self.axis];
                }
                new_shape.remove(self.axis);
                new_stride.remove(self.axis);

                let mut buffer_count: Vec<usize> = vec![0; self.axis + 1];

                for _i in 0..self.indices {
                    // Calculating offset on each iteration works like a counter, where each digit is an element
                    // in an array/vector with a base corresponding to the shape at the index of the digit.
                    // In the 'units' place, the 'base' is the stride at the axis of iteration.
                    // These 'digits' are maintained in buffer_count

                    let mut curr_axis = self.axis as isize;
                    data_vec.push(
                        *self
                            .data_buffer
                            .const_ptr()
                            .offset(self.iter_count * self.stride[self.axis] as isize + ptr_offset),
                    );

                    buffer_count[curr_axis as usize] += 1;
                    ptr_offset += 1;
                    while curr_axis >= 0
                        && ((curr_axis == self.axis as isize
                            && buffer_count[curr_axis as usize] == self.stride[self.axis])
                            || (curr_axis != self.axis as isize
                                && buffer_count[curr_axis as usize]
                                    == self.shape[curr_axis as usize]))
                    {
                        buffer_count[curr_axis as usize] = 0;
                        curr_axis -= 1;

                        if curr_axis < 0 {
                            break;
                        }
                        buffer_count[curr_axis as usize] += 1;
                        ptr_offset = (buffer_count[curr_axis as usize]
                            * self.stride[curr_axis as usize])
                            as isize;
                    }
                }

                let data_buffer = DataView::from_vec_ref(data_vec.clone(), 0, data_vec.len());

                self.iter_count += 1;

                Some(TensorView {
                    data: data_buffer,
                    shape: new_shape.clone(),
                    stride: new_stride.clone(),
                    ndims: new_shape.len(),
                })
            },
        }
    }
}
