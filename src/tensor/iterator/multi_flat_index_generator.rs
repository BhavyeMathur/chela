use crate::tensor::MAX_DIMS;

#[non_exhaustive]
pub struct MultiFlatIndexGenerator<const N: usize>
{
    ndims: usize,
    shape: [usize; MAX_DIMS],
    strides: [[usize; MAX_DIMS]; N],

    size: usize,
    iterator_index: usize,

    indices: [usize; MAX_DIMS], // current index along each dimension
    flat_indices: [usize; N],
}

impl<const N: usize> MultiFlatIndexGenerator<N> {
    pub(in crate::tensor) fn from<const M: usize>(shape: &[usize], strides: &[[usize; M]; N]) -> Self {
        let ndims = shape.len();
        assert!(M >= ndims);

        let size = shape.iter().product();

        let mut new_shape = [0; MAX_DIMS];
        new_shape[0..ndims].copy_from_slice(&shape);

        let mut new_strides = [[0; MAX_DIMS]; N];
        for (stride, new_stride) in strides.iter().zip(new_strides.iter_mut()) {
            new_stride[0..ndims].copy_from_slice(&stride[0..ndims]);
        }

        Self {
            ndims,
            shape: new_shape.clone(),
            strides: new_strides,
            size,
            iterator_index: 0,
            indices: new_shape,
            flat_indices: [0; N],
        }
    }

    #[inline]
    pub(crate) unsafe fn cur_indices(&mut self) -> &[usize; N] {
        &self.flat_indices
    }

    // SAFETY: this function does not increment self.iterator_index
    #[inline]
    pub(crate) unsafe fn increment_flat_indices(&mut self) {
        let mut idim = self.ndims;

        while idim != 0 {
            idim -= 1;

            unsafe {
                let idx = self.indices.get_unchecked_mut(idim);
                let dimension = *self.shape.get_unchecked(idim);
                *idx -= 1;

                if *idx != 0 {
                    for (flat_index, stride) in self.flat_indices.iter_mut().zip(self.strides.iter()) {
                        *flat_index += *stride.get_unchecked(idim);
                    }
                    return;
                }

                *idx = dimension; // reset this dimension and carry over to the next
                for (flat_index, stride) in self.flat_indices.iter_mut().zip(self.strides.iter()) {
                    *flat_index -= *stride.get_unchecked(idim) * (dimension - 1);
                }
            }
        }
    }
}

impl<const N: usize> Iterator for MultiFlatIndexGenerator<N> {
    type Item = [usize; N];

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.iterator_index == self.size {
            return None;
        }

        let return_indices = self.flat_indices.clone();
        unsafe { self.increment_flat_indices() };
        self.iterator_index += 1;

        Some(return_indices)
    }
}

impl<const N: usize> Clone for MultiFlatIndexGenerator<N> {
    fn clone(&self) -> Self {
        Self {
            ndims: self.ndims,
            shape: self.shape.clone(),
            strides: self.strides.clone(),

            size: self.size,
            iterator_index: self.iterator_index,

            indices: self.indices.clone(),
            flat_indices: self.flat_indices,
        }
    }
}
