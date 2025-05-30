use crate::dtype::{NumericDataType, RawDataType};
use crate::flat_index_generator::FlatIndexGenerator;
use crate::iterator::collapse_contiguous::collapse_to_uniform_stride;
use crate::util::to_vec::ToVec;
use crate::{AxisType, FloatDataType, NdArray, StridedMemory};
use num::{NumCast};
use std::collections::VecDeque;

/// Returns a tuple `(output_shape, map_stride)`
///
/// - `output_shape` is the shape of the output ndarray after the reduction operation
///
/// - `map_stride` maps a flat iteration over the input ndarray to iteration over the output ndarray.
///   For example, if the reduce operation is addition, `reduce` iterates through the input ndarray
///   element-by-element and `map_stride` describes iteration over the output ndarray
///   to add each element to the correct location.
///   It should now make sense why map_stride contains zeros on every reduced axis
fn reduced_shape_and_stride(axes: &[isize], shape: &[usize]) -> (Vec<usize>, Vec<usize>) {
    let ndims = shape.len();
    let mut axis_mask = vec![false; ndims];

    for &axis in axes.iter() {
        let axis = axis.get_absolute(ndims);
        if axis_mask[axis] {
            panic!("duplicate axes specified");
        }
        axis_mask[axis] = true;
    }

    let mut new_stride = VecDeque::with_capacity(ndims);
    let mut new_shape = VecDeque::with_capacity(ndims - axes.len());

    let mut stride = 1;
    for axis in (0..ndims).rev() {
        if axis_mask[axis] {
            new_stride.push_front(0);
        } else {
            new_stride.push_front(stride);
            new_shape.push_front(shape[axis]);
            stride *= shape[axis];
        }
    }

    (Vec::from(new_shape), Vec::from(new_stride))
}

impl<T: RawDataType> NdArray<'_, T> {
    /// Reduces the elements of a contiguous ndarray into a scalar using the specified function.
    ///
    /// # Safety
    /// - Ensure that the underlying ndarray is contiguous in memory with a stride of 1.
    ///
    /// # Parameters
    /// - `func`: A closure or function that takes two arguments (the next value to be reduced
    ///   and the value of the accumulator) and returns a reduction of both.
    ///   For example, when the reduction operation is addition, `|src, acc| src + acc`
    /// - `default`: The initial value used as the accumulator for the reduction.
    unsafe fn reduce_contiguous<'a, 'b>(&'a self, func: impl Fn(T, T) -> T, default: T) -> NdArray<'b, T> {
        let mut output = default;

        let mut src = self.ptr();
        for _ in 0..self.len {
            output = func(*src, output);
            src = src.add(1);  // TODO we can generalise this to other strides
        }

        NdArray::scalar(output)
    }
}

impl<T: RawDataType> NdArray<'_, T> {
    pub fn reduce_along<'a, 'b>(&'a self, func: impl Fn(T, T) -> T, axes: impl ToVec<isize>, default: T) -> NdArray<'b, T> {
        let (out_shape, map_stride) = reduced_shape_and_stride(&axes.to_vec(), &self.shape);
        let (map_shape, map_stride) = collapse_to_uniform_stride(&self.shape, &map_stride);

        let mut output = vec![default; out_shape.iter().product()];

        let mut dst_indices = FlatIndexGenerator::from(&map_shape, &map_stride);
        let dst: *mut T = output.as_mut_ptr();

        for el in self.flatiter() {
            unsafe {
                let dst_i = dst_indices.next().unwrap();
                let dst_ptr = dst.add(dst_i);
                *dst_ptr = func(el, *dst_ptr);
            }
        }

        unsafe { NdArray::from_contiguous_owned_buffer(out_shape, output) }
    }

    pub fn reduce<'a, 'b>(&'a self, func: impl Fn(T, T) -> T, default: T) -> NdArray<'b, T> {
        if self.is_contiguous() {
            return unsafe { self.reduce_contiguous(func, default) };
        }

        let mut output = default;

        for el in self.flatiter() {
            output = func(el, output);
        }

        NdArray::scalar(output)
    }
}

impl<T: NumericDataType> NdArray<'_, T> {
    pub fn sum<'a, 'b>(&'a self) -> NdArray<'b, T> {
        self.reduce(|val, acc| acc + val, T::zero())
    }

    pub fn sum_along<'a, 'b>(&'a self, axes: impl ToVec<isize>) -> NdArray<'b, T> {
        self.reduce_along(|val, acc| acc + val, axes, T::zero())
    }

    pub fn product<'a, 'b>(&'a self) -> NdArray<'b, T> {
        self.reduce(|val, acc| acc * val, T::one())
    }

    pub fn product_along<'a, 'b>(&'a self, axes: impl ToVec<isize>) -> NdArray<'b, T> {
        self.reduce_along(|val, acc| acc * val, axes, T::one())
    }

    pub fn max<'a, 'b>(&'a self) -> NdArray<'b, T> {
        self.reduce(partial_max, T::min_value())
    }

    pub fn min<'a, 'b>(&'a self) -> NdArray<'b, T> {
        self.reduce(partial_min, T::max_value())
    }

    pub fn max_along<'a, 'b>(&'a self, axes: impl ToVec<isize>) -> NdArray<'b, T> {
        self.reduce_along(partial_max, axes, T::min_value())
    }

    pub fn min_along<'a, 'b>(&'a self, axes: impl ToVec<isize>) -> NdArray<'b, T> {
        self.reduce_along(partial_min, axes, T::max_value())
    }

    pub fn max_magnitude<'a, 'b>(&'a self) -> NdArray<'b, T> {
        self.reduce(partial_max_magnitude, T::zero())
    }

    pub fn min_magnitude<'a, 'b>(&'a self) -> NdArray<'b, T> {
        self.reduce(partial_min_magnitude, T::zero())
    }

    pub fn max_magnitude_along<'a, 'b>(&'a self, axes: impl ToVec<isize>) -> NdArray<'b, T> {
        self.reduce_along(partial_max_magnitude, axes, T::zero())
    }

    pub fn min_magnitude_along<'a, 'b>(&'a self, axes: impl ToVec<isize>) -> NdArray<'b, T> {
        self.reduce_along(partial_min_magnitude, axes, T::zero())
    }
}

// #[cfg(not(use_apple_vdsp))]
// impl NdArrayNumericReduce<f32> for NdArray<'_, f32> {}
//
// #[cfg(not(use_apple_vdsp))]
// impl NdArrayNumericReduce<f64> for NdArray<'_, f64> {}
// TODO
// 
// #[cfg(use_apple_vdsp)]
// impl NdArrayNumericReduce<f32> for NdArray<'_, f32> {
//     fn sum<'a, 'b>(&'a self) -> NdArray<'b, f32> {
//         match self.has_uniform_stride() {
//             None => { self.reduce(|val, acc| acc + val, 0.0) }
//             Some(stride) => {
//                 let mut output = 0.0;
//                 unsafe { vDSP_sve(self.ptr(), stride as isize, std::ptr::addr_of_mut!(output), self.size() as isize); }
//                 NdArray::scalar_requires_grad(output, self.requires_grad())
//             }
//         }
//     }
//
//     fn max<'a, 'b>(&'a self) -> NdArray<'b, f32> {
//         match self.has_uniform_stride() {
//             None => { self.reduce(partial_max, f32::min_value()) }
//             Some(stride) => {
//                 let mut output = 0.0;
//                 unsafe { vDSP_maxv(self.ptr(), stride as isize, std::ptr::addr_of_mut!(output), self.size() as isize); }
//                 NdArray::scalar_requires_grad(output, self.requires_grad())
//             }
//         }
//     }
//
//     fn min<'a, 'b>(&'a self) -> NdArray<'b, f32> {
//         match self.has_uniform_stride() {
//             None => { self.reduce(partial_min, f32::max_value()) }
//             Some(stride) => {
//                 let mut output = 0.0;
//                 unsafe { vDSP_minv(self.ptr(), stride as isize, std::ptr::addr_of_mut!(output), self.size() as isize); }
//                 NdArray::scalar_requires_grad(output, self.requires_grad())
//             }
//         }
//     }
//
//     fn max_magnitude<'a, 'b>(&'a self) -> NdArray<'b, f32> {
//         match self.has_uniform_stride() {
//             None => { self.reduce(partial_max_magnitude, 0.0) }
//             Some(stride) => {
//                 let mut output = 0.0;
//                 unsafe { vDSP_maxmgv(self.ptr(), stride as isize, std::ptr::addr_of_mut!(output), self.size() as isize); }
//                 NdArray::scalar_requires_grad(output, self.requires_grad())
//             }
//         }
//     }
//
//     fn min_magnitude<'a, 'b>(&'a self) -> NdArray<'b, f32> {
//         match self.has_uniform_stride() {
//             None => { self.reduce(partial_min_magnitude, 0.0) }
//             Some(stride) => {
//                 let mut output = 0.0;
//                 unsafe { vDSP_minmgv(self.ptr(), stride as isize, std::ptr::addr_of_mut!(output), self.size() as isize); }
//                 NdArray::scalar_requires_grad(output, self.requires_grad())
//             }
//         }
//     }
// }
//
// #[cfg(use_apple_vdsp)]
// impl NdArrayNumericReduce<f64> for NdArray<'_, f64> {
//     fn sum<'a, 'b>(&'a self) -> NdArray<'b, f64> {
//         match self.has_uniform_stride() {
//             None => { self.reduce(|val, acc| acc + val, 0.0) }
//             Some(stride) => {
//                 let mut output = 0.0;
//                 unsafe { vDSP_sveD(self.ptr(), stride as isize, std::ptr::addr_of_mut!(output), self.size() as isize); }
//                 NdArray::scalar_requires_grad(output, self.requires_grad())
//             }
//         }
//     }
//
//     fn max<'a, 'b>(&'a self) -> NdArray<'b, f64> {
//         match self.has_uniform_stride() {
//             None => { self.reduce(partial_max, f64::min_value()) }
//             Some(stride) => {
//                 let mut output = 0.0;
//                 unsafe { vDSP_maxvD(self.ptr(), stride as isize, std::ptr::addr_of_mut!(output), self.size() as isize); }
//                 NdArray::scalar_requires_grad(output, self.requires_grad())
//             }
//         }
//     }
//
//     fn min<'a, 'b>(&'a self) -> NdArray<'b, f64> {
//         match self.has_uniform_stride() {
//             None => { self.reduce(partial_min, f64::max_value()) }
//             Some(stride) => {
//                 let mut output = 0.0;
//                 unsafe { vDSP_minvD(self.ptr(), stride as isize, std::ptr::addr_of_mut!(output), self.size() as isize); }
//                 NdArray::scalar_requires_grad(output, self.requires_grad())
//             }
//         }
//     }
//
//     fn max_magnitude<'a, 'b>(&'a self) -> NdArray<'b, f64> {
//         match self.has_uniform_stride() {
//             None => { self.reduce(partial_max_magnitude, 0.0) }
//             Some(stride) => {
//                 let mut output = 0.0;
//                 unsafe { vDSP_maxmgvD(self.ptr(), stride as isize, std::ptr::addr_of_mut!(output), self.size() as isize); }
//                 NdArray::scalar_requires_grad(output, self.requires_grad())
//             }
//         }
//     }
//
//     fn min_magnitude<'a, 'b>(&'a self) -> NdArray<'b, f64> {
//         match self.has_uniform_stride() {
//             None => { self.reduce(partial_min_magnitude, 0.0) }
//             Some(stride) => {
//                 let mut output = 0.0;
//                 unsafe { vDSP_minmgvD(self.ptr(), stride as isize, std::ptr::addr_of_mut!(output), self.size() as isize); }
//                 NdArray::scalar_requires_grad(output, self.requires_grad())
//             }
//         }
//     }
// }


impl<T: FloatDataType> NdArray<'_, T> {
    pub fn mean<'a, 'b>(&'a self) -> NdArray<'b, T> {
        let n: T = NumCast::from(self.size()).unwrap();
        self.sum() / n
    }

    pub fn mean_along<'a, 'b>(&'a self, axes: impl ToVec<isize>) -> NdArray<'b, T> {
        let axes = axes.to_vec();

        let mut n = 1;
        for &axis in axes.iter() {
            assert!(axis >= 0, "negative axes are not currently supported");
            n *= self.shape[axis as usize];
        }
        
        let n: T = NumCast::from(n).unwrap();
        self.sum_along(axes) / n
    }
}

#[cfg(test)]
mod tests {
    use crate::reduce::reduced_shape_and_stride;

    #[test]
    fn test_reduce_shape_and_stride() {
        let shape = vec![3, 2];

        let correct_shape = vec![3];
        let correct_stride = vec![1, 0];
        let (new_shape, new_stride) = reduced_shape_and_stride(&vec![1], &shape);
        assert_eq!(new_shape, correct_shape);
        assert_eq!(new_stride, correct_stride);

        let shape = vec![4, 2, 3];

        let correct_shape = vec![2, 3];
        let correct_stride = vec![0, 3, 1];
        let (new_shape, new_stride) = reduced_shape_and_stride(&vec![0], &shape);
        assert_eq!(new_shape, correct_shape);
        assert_eq!(new_stride, correct_stride);

        let correct_shape = vec![4, 3];
        let correct_stride = vec![3, 0, 1];
        let (new_shape, new_stride) = reduced_shape_and_stride(&vec![1], &shape);
        assert_eq!(new_shape, correct_shape);
        assert_eq!(new_stride, correct_stride);

        let correct_shape = vec![4, 2];
        let correct_stride = vec![2, 1, 0];
        let (new_shape, new_stride) = reduced_shape_and_stride(&vec![2], &shape);
        assert_eq!(new_shape, correct_shape);
        assert_eq!(new_stride, correct_stride);

        let correct_shape = vec![3];
        let correct_stride = vec![0, 0, 1];
        let (new_shape, new_stride) = reduced_shape_and_stride(&vec![0, 1], &shape);
        assert_eq!(new_shape, correct_shape);
        assert_eq!(new_stride, correct_stride);

        let correct_shape = vec![2];
        let correct_stride = vec![0, 1, 0];
        let (new_shape, new_stride) = reduced_shape_and_stride(&vec![0, 2], &shape);
        assert_eq!(new_shape, correct_shape);
        assert_eq!(new_stride, correct_stride);

        let correct_shape = vec![4];
        let correct_stride = vec![1, 0, 0];
        let (new_shape, new_stride) = reduced_shape_and_stride(&vec![1, 2], &shape);
        assert_eq!(new_shape, correct_shape);
        assert_eq!(new_stride, correct_stride);
    }
}

fn partial_max<T: NumericDataType>(a: T, b: T) -> T {
    if a.partial_cmp(&b) == Some(std::cmp::Ordering::Greater) { a } else { b }
}

fn partial_min<T: NumericDataType>(a: T, b: T) -> T {
    if a.partial_cmp(&b) == Some(std::cmp::Ordering::Less) { a } else { b }
}

fn partial_max_magnitude<T: NumericDataType>(val: T, acc: T) -> T {
    let val = val.abs();
    if acc.partial_cmp(&val) == Some(std::cmp::Ordering::Greater) { acc } else { val }
}

fn partial_min_magnitude<T: NumericDataType>(val: T, acc: T) -> T {
    let val = val.abs();
    if acc.partial_cmp(&val) == Some(std::cmp::Ordering::Less) { acc } else { val }
}
