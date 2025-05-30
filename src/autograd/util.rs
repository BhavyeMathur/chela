use crate::broadcast::get_broadcasted_axes;
use crate::{FloatDataType, NdArray, Reshape, StridedMemory};


pub(super) fn reduce_gradient<'a, T: FloatDataType>(grad: &'a NdArray<'a, T>,
                                                    original_shape: &[usize]) -> NdArray<'a, T> {
    if grad.shape() == original_shape {
        return grad.view()
    }

    let axes = get_broadcasted_axes(grad.shape(), original_shape);
    let grad = grad.sum_along(axes);

    grad.reshape(original_shape)
}
