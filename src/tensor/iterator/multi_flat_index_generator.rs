use crate::tensor::{MAX_ARGS, MAX_DIMS};
use crate::NumericDataType;

#[non_exhaustive]
pub struct MultiFlatIndexGenerator
{
    ndims: usize,
    nops: usize,
    shape: [usize; MAX_DIMS],
    strides: [[usize; MAX_ARGS]; MAX_DIMS],

    size: usize,
    iterator_index: usize,

    indices: [usize; MAX_DIMS], // current index along each dimension
    flat_indices: [usize; MAX_ARGS],
}

impl MultiFlatIndexGenerator {
    pub(crate) fn find_best_axis_ordering<const OPERANDS: usize>(nops: usize,
                                                                 ndims: usize,
                                                                 strides: &[[usize; OPERANDS]])
                                                                 -> Option<Vec<usize>> {
        assert!(ndims <= MAX_DIMS);

        let mut permuted = false;
        let mut best_axis_ordering = Vec::with_capacity(ndims);
        for i in 0..ndims {
            best_axis_ordering.push(i)
        }

        for ax_i0 in 1..ndims {
            let mut ax_ipos = ax_i0;
            let ax_j0 = best_axis_ordering[ax_i0];
            let strides0 = strides[ax_j0];

            for ax_i1 in (0..ax_i0).rev() {
                let mut ambiguous = true;
                let mut should_swap = false;

                let ax_j1 = best_axis_ordering[ax_i1];
                let strides1 = strides[ax_j1];

                for iop in 0..nops {
                    if strides0[iop] != 0 && strides1[iop] != 0 {
                        if strides1[iop].abs() <= strides0[iop].abs() {
                            should_swap = false;
                        } else if ambiguous {
                            should_swap = true;
                        }

                        ambiguous = false;
                    }
                }

                if !ambiguous {
                    if should_swap {
                        ax_ipos = ax_i1;
                    } else {
                        break;
                    }
                }
            }

            if ax_ipos != ax_i0 {
                for ax_i1 in ((ax_ipos + 1)..=ax_i0).rev() {
                    best_axis_ordering[ax_i1] = best_axis_ordering[ax_i1 - 1];
                }
                best_axis_ordering[ax_ipos] = ax_j0;
                permuted = true;
            }
        }

        if permuted { Some(best_axis_ordering) } else { None }
    }

    pub(crate) fn from<const OPERANDS: usize, const DIMS: usize>(nops: usize,
                                                                 shape: &[usize],
                                                                 strides: &[[usize; OPERANDS]; DIMS])
                                                                 -> Self {
        let ndims = shape.len();

        assert!(OPERANDS <= MAX_ARGS);
        assert!(DIMS <= MAX_DIMS);
        assert!(nops <= OPERANDS);
        assert!(ndims <= DIMS);

        let size = shape.iter().product();

        let mut new_shape = [0; MAX_DIMS];
        new_shape[0..ndims].copy_from_slice(&shape);

        let mut new_strides = [[0; MAX_ARGS]; MAX_DIMS];
        for j in 0..ndims {
            new_strides[j][0..nops].copy_from_slice(&strides[j][0..nops]);
        }

        Self {
            ndims,
            nops,
            shape: new_shape.clone(),
            strides: new_strides,
            size,
            iterator_index: 0,
            indices: new_shape,
            flat_indices: [0; MAX_ARGS],
        }
    }

    #[inline]
    pub(crate) unsafe fn cur_indices(&mut self) -> &[usize; MAX_ARGS] {
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
                let strides = self.strides.get_unchecked(idim);
                *idx -= 1;

                if *idx != 0 {
                    for i in 0..self.nops {
                        *self.flat_indices.get_unchecked_mut(i) += strides.get_unchecked(i);
                    }
                    return;
                }

                *idx = dimension; // reset this dimension and carry over to the next
                for i in 0..self.nops {
                    *self.flat_indices.get_unchecked_mut(i) -= strides.get_unchecked(i) * (dimension - 1);
                }
            }
        }
    }
}

impl Iterator for MultiFlatIndexGenerator {
    type Item = [usize; MAX_ARGS];

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

impl Clone for MultiFlatIndexGenerator {
    fn clone(&self) -> Self {
        Self {
            ndims: self.ndims,
            nops: self.nops,
            shape: self.shape.clone(),
            strides: self.strides.clone(),

            size: self.size,
            iterator_index: self.iterator_index,

            indices: self.indices.clone(),
            flat_indices: self.flat_indices,
        }
    }
}
