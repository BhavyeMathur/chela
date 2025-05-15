use crate::iterator::collapse_contiguous::collapse_to_uniform_stride;
use crate::tensor::MAX_DIMS;

#[non_exhaustive]
pub struct FlatIndexGenerator
{
    ndims: usize,
    shape: [usize; MAX_DIMS],
    stride: [usize; MAX_DIMS],

    size: usize,
    iterator_index: usize,

    indices: [usize; MAX_DIMS], // current index along each dimension
    flat_index: usize,
}

impl FlatIndexGenerator {
    pub(in crate::tensor) fn from(shape: &[usize], stride: &[usize]) -> Self {
        let (shape, stride) = collapse_to_uniform_stride(shape, stride);
        let ndims = shape.len();
        let size = shape.iter().product();

        let mut new_shape = [0; MAX_DIMS];
        let mut new_stride = [0; MAX_DIMS];

        new_shape[0..ndims].copy_from_slice(&shape);
        new_stride[0..ndims].copy_from_slice(&stride);

        Self {
            ndims,
            shape: new_shape,
            stride: new_stride,
            size,
            iterator_index: 0,
            indices: [0; MAX_DIMS],
            flat_index: 0,
        }
    }

    #[inline(always)]
    pub fn advance_by(&mut self, mut n: usize) {
        let remaining = self.size - self.iterator_index;
        n = n.min(remaining);

        if n == 0 {
            return;
        }
        self.iterator_index += n;

        let mut carry = n;
        for i in (0..self.ndims).rev() {
            let dim = self.shape[i];
            let idx = &mut self.indices[i];

            let total = *idx + carry;
            *idx = total % dim;
            carry = total / dim;

            self.flat_index += self.stride[i] * (*idx - self.indices[i]);
        }
    }
}

impl Iterator for FlatIndexGenerator {
    type Item = usize;

    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        if self.iterator_index == self.size {
            return None;
        }

        let return_index = self.flat_index;

        let mut i = self.ndims;
        while i > 0 {
            i -= 1;

            unsafe {
                let idx = self.indices.get_unchecked_mut(i);
                *idx += 1;

                if *idx < *self.shape.get_unchecked(i) {
                    self.flat_index += *self.stride.get_unchecked(i);
                    break;
                }

                *idx = 0; // reset this dimension and carry over to the next
                self.flat_index -= *self.stride.get_unchecked(i) * (*self.shape.get_unchecked(i) - 1);
            }
        }

        self.iterator_index += 1;
        Some(return_index)
    }
}

impl Clone for FlatIndexGenerator {
    fn clone(&self) -> Self {
        Self {
            ndims: self.ndims,
            shape: self.shape.clone(),
            stride: self.stride.clone(),

            size: self.size,
            iterator_index: self.iterator_index,

            indices: self.indices.clone(),
            flat_index: self.flat_index,
        }
    }
}
