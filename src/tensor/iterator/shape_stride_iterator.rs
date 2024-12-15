use crate::iterator::collapse_contiguous::collapse_contiguous;

#[non_exhaustive]
pub struct ShapeStrideRange
{
    shape: Vec<usize>,
    stride: Vec<usize>,
    ndims: usize,

    size: usize,
    iterator_index: usize,

    indices: Vec<usize>, // current index along each dimension
    flat_index: usize,
}

impl ShapeStrideRange {
    pub(super) fn from(shape: &Vec<usize>, stride: &Vec<usize>) -> Self {
        let (shape, stride) = collapse_contiguous(shape, stride);
        let ndims = shape.len();
        let size = shape.iter().product();

        Self {
            shape,
            stride,
            ndims,
            size,
            iterator_index: 0,
            indices: vec![0; ndims],
            flat_index: 0,
        }
    }
}

impl Iterator for ShapeStrideRange {
    type Item = isize;

    fn next(&mut self) -> Option<Self::Item> {
        if self.iterator_index == self.size {
            return None;
        }

        let return_index = self.flat_index as isize;

        for i in (0..self.ndims).rev() {
            self.indices[i] += 1;

            if self.indices[i] < self.shape[i] {
                self.flat_index += self.stride[i];
                break;
            }

            self.flat_index -= self.stride[i] * (self.shape[i] - 1);
            self.indices[i] = 0; // reset this dimension and carry over to the next
        }

        self.iterator_index += 1;
        Some(return_index)
    }
}