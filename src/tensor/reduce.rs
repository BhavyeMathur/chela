use crate::dtype::{NumericDataType, RawDataType};
use crate::flat_index_generator::FlatIndexGenerator;
use crate::iterator::collapse_contiguous::collapse_to_uniform_stride;
use crate::traits::to_vec::ToVec;
use crate::Tensor;
use std::cmp::{max, min};
use std::collections::VecDeque;
use std::ops::Div;

// returns a tuple (output_shape, map_stride)
// output_shape is simply the shape of the output tensor after the reduction operation
//
// map_stride maps a flat iteration over the input tensor to iteration over the output tensor.
// for example, if the reduce operation is addition,
// the reduce() function iterates through the input tensor element-by-element
// map_stride tells reduce() how to iterate over the output tensor
// to add each element to the correct location.
// it should now make sense why map_stride contains 0s on every reduced axis
fn reduced_shape_and_stride(axes: &[usize], shape: &[usize]) -> (Vec<usize>, Vec<usize>) {
    let ndims = shape.len();
    let mut axis_mask = vec![false; ndims];

    for &axis in axes.iter() {
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


impl<T: RawDataType> Tensor<'_, T> {
    pub fn reduce_along(&self, func: impl Fn(T, T) -> T, axes: impl ToVec<usize>, default: T) -> Tensor<T> {
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

        unsafe { Tensor::from_contiguous_owned_buffer(out_shape, output) }
    }

    unsafe fn reduce_contiguous(&self, func: impl Fn(T, T) -> T, default: T) -> Tensor<T> {
        let mut output = default;

        let mut src: *mut T = self.ptr.as_ptr();
        for _ in 0..self.len {
            output = func(*src, output);
            src = src.add(1);
        }

        Tensor::scalar(output)
    }

    pub fn reduce(&self, func: impl Fn(T, T) -> T, default: T) -> Tensor<T> {
        if self.is_contiguous() {
            return unsafe { self.reduce_contiguous(func, default) };
        }

        let mut output = default;

        for el in self.flatiter() {
            output = func(el, output);
        }

        Tensor::scalar(output)
    }
}

fn reduce_sum<T: NumericDataType>(value: T, accumulator: T) -> T {
    accumulator + value
}

fn reduce_product<T: NumericDataType>(value: T, accumulator: T) -> T {
    accumulator * value
}

fn reduce_min<T: NumericDataType + Ord>(value: T, accumulator: T) -> T {
    min(accumulator, value)
}

fn reduce_max<T: NumericDataType + Ord>(value: T, accumulator: T) -> T {
    max(accumulator, value)
}

impl<T: NumericDataType + num::Zero> Tensor<'_, T> {
    pub fn sum(&self) -> Tensor<T> {
        self.reduce(reduce_sum, T::zero())
    }

    pub fn sum_along(&self, axes: impl ToVec<usize>) -> Tensor<T> {
        self.reduce_along(reduce_sum, axes, T::zero())
    }
}

impl<T: NumericDataType + num::One> Tensor<'_, T> {
    pub fn product(&self) -> Tensor<T> {
        self.reduce(reduce_product, T::one())
    }

    pub fn product_along(&self, axes: impl ToVec<usize>) -> Tensor<T> {
        self.reduce_along(reduce_product, axes, T::one())
    }
}

impl<T: NumericDataType + num::Zero> Tensor<'_, T>
where
        for<'a> Tensor<'a, T>: Div<T::AsFloatType>,
{
    pub fn mean(&self) -> <Tensor<T> as Div<T::AsFloatType>>::Output {
        self.reduce(reduce_sum, T::zero()) / (self.size() as f32).into()
    }

    pub fn mean_along(&self, axes: impl ToVec<usize>) -> <Tensor<T> as Div<T::AsFloatType>>::Output {
        let axes = axes.to_vec();

        let mut n = 1;
        for &axis in axes.iter() {
            n *= self.shape[axis];
        }
        let n: T::AsFloatType = (n as f32).into();

        self.reduce_along(reduce_sum, axes, T::zero()) / n
    }
}

impl<T: NumericDataType + num::Bounded + Ord> Tensor<'_, T> {
    pub fn max(&self) -> Tensor<T> {
        self.reduce(reduce_max, T::min_value())
    }

    pub fn min(&self) -> Tensor<T> {
        self.reduce(reduce_min, T::max_value())
    }

    pub fn max_along(&self, axes: impl ToVec<usize>) -> Tensor<T> {
        self.reduce_along(reduce_max, axes, T::min_value())
    }

    pub fn min_along(&self, axes: impl ToVec<usize>) -> Tensor<T> {
        self.reduce_along(reduce_min, axes, T::max_value())
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
