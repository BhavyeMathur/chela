use crate::dtype::RawDataType;
use crate::traits::to_vec::ToVec;
use crate::Tensor;

impl<'a, T: RawDataType> Tensor<'a, T> {
    pub fn broadcast_to(&'a self, shape: impl ToVec<usize>) -> Tensor<'a, T> {
        let shape = shape.to_vec();

        let broadcast_shape = broadcast_shape(&self.shape, shape);
        let broadcast_stride = broadcast_stride(&self.stride, &broadcast_shape, &self.shape);

        unsafe { self.reshaped_view(broadcast_shape, broadcast_stride) }
    }
}

fn pad<T: Copy>(arr: &[T], value: T, n: usize) -> Vec<T> {
    let mut new_arr = Vec::with_capacity(arr.len() + n);

    for _ in 0..n {
        new_arr.push(value);
    }
    new_arr.extend(arr);
    new_arr
}

fn pad_dimensions(shape: &[usize], stride: &[usize], ndims: usize) -> (Vec<usize>, Vec<usize>) {
    let n = ndims - shape.len();
    let shape = pad(shape, 1, n);
    let stride = pad(stride, 0, n);

    (shape, stride)
}

fn broadcast_shape(shape: &[usize], to: impl ToVec<usize>) -> Vec<usize> {
    let to = to.to_vec();
    let ndims = to.len();

    if ndims < shape.len() {
        panic!("cannot broadcast to fewer dimensions")
    }


    // TODO we can just copy shape and return that since broadcast_shape == to at the end
    // just need to check that the broadcasting is compatible

    let mut broadcast_shape = pad(shape, 1, ndims - shape.len());

    for axis in 0..ndims {
        if to[axis] == broadcast_shape[axis] {
            continue;
        }

        if broadcast_shape[axis] != 1 {
            panic!("broadcast is not compatible with the desired shape");
        }

        broadcast_shape[axis] = to[axis];
    }

    broadcast_shape
}

fn broadcast_stride(stride: &[usize], broadcast_shape: &[usize], original_shape: &[usize]) -> Vec<usize> {
    let ndims = broadcast_shape.len();
    let original_first_axis = ndims - original_shape.len();

    if original_first_axis < 0 {
        panic!("cannot broadcast to fewer dimensions")
    }

    let mut broadcast_stride = Vec::with_capacity(ndims);

    for _ in 0..original_first_axis {
        broadcast_stride.push(0);  // new dimensions get a zero stride
    }

    for axis in original_first_axis..ndims {
        let original_axis_length = original_shape[axis - original_first_axis];

        if original_axis_length == 1 {
            broadcast_stride.push(0);
        } else if original_axis_length == broadcast_shape[axis] {
            broadcast_stride.push(stride[axis - original_first_axis]);
        } else {
            panic!("broadcast is not compatible with the desired shape");
        }
    }

    broadcast_stride
}

// TODO tests for broadcast_stride()

fn broadcast_shapes(first: &[usize], second: &[usize]) -> Vec<usize> {
    todo!()
}
