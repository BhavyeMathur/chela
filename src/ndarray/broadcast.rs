use crate::dtype::RawDataType;
use crate::reshape::{ReshapeImpl};
use crate::ndarray::flags::NdArrayFlags;
use crate::util::functions::pad;
use crate::NdArray;

impl<'a, T: RawDataType> NdArray<'a, T> {
    /// Broadcasts the tensor to the specified shape.
    ///
    /// This method returns a *readonly* view of the tensor with the desired shape.
    /// Broadcasting is done by left-padding the tensor's shape with ones until they reach the
    /// desired dimension. Then, any axes with length 1 are repeated to match the target shape.
    ///
    /// For example, suppose the tensor's shape is `[2, 3]` and the broadcast shape is `[3, 2, 3]`.
    /// Then the tensor's shape becomes `[1, 2, 3]` after padding and `[3, 2, 3]` after repeating
    /// the first axis.
    ///
    /// # Panics
    /// This method panics if the target shape is incompatible with the tensor.
    ///
    /// - If `shape.len()` is less than the dimensionality of the tensor.
    /// - If a dimension in `shape` does not equal the corresponding dimension in the tensor's `shape`
    ///   and cannot be broadcasted (i.e., it is not 1 or does not match).
    ///
    /// # Example
    ///
    /// ```rust
    /// # use chela::*;
    /// let tensor = NdArray::from([1, 2, 3]);  // shape is [3]
    /// let broadcasted_tensor = tensor.broadcast_to(&[2, 3]);
    ///
    /// assert_eq!(broadcasted_tensor.shape(), &[2, 3]);
    /// ```
    pub fn broadcast_to(&'a self, shape: &[usize]) -> NdArray<'a, T> {
        let broadcast_shape = broadcast_shape(&self.shape, shape);
        let broadcast_stride = broadcast_stride(&self.stride, &broadcast_shape, &self.shape);

        let mut result = unsafe { self.reshaped_view(broadcast_shape, broadcast_stride) };
        result.flags -= NdArrayFlags::Writeable;
        result
    }
}

/// Adjusts `shape` and `stride` to match an `ndims`-dimensional view of the tensor
///
/// This is done by left-padding `shape` with ones and `stride` with zeros until they reach
/// the desired dimension.
///
/// # Panics
/// - If `shape.len() > ndims`
///
/// # Example
/// ```ignore
/// let shape = vec![2, 3];
/// let stride = vec![3, 1];
/// let ndims = 4;
///
/// let (padded_shape, padded_stride) = pad_dimensions(&shape, &stride, ndims);
///
/// assert_eq!(padded_shape, vec![1, 1, 2, 3]);
/// assert_eq!(padded_stride, vec![0, 0, 3, 1]);
/// ```
fn pad_dimensions(shape: &[usize], stride: &[usize], ndims: usize) -> (Vec<usize>, Vec<usize>) {
    let n = ndims - shape.len();
    let shape = pad(shape, 1, n);
    let stride = pad(stride, 0, n);

    (shape, stride)
}

/// Checks if broadcasting a shape to another is possible. Panics otherwise.
///
/// Broadcasting is done by left-padding the tensor's shape with ones until they reach the
/// desired dimension. Then, any axes with length 1 are repeated to match the target shape.
///
/// For example, suppose `shape` is `[2, 3]` and `to` is `[3, 2, 3]`.
/// Then `shape` becomes `[1, 2, 3]` after padding and `[3, 2, 3]` after repeating the first axis.
///
/// # Panics
/// - If the number of dimensions in `to` is less than the number of dimensions in `shape`.
/// - If a dimension in `shape` does not equal the corresponding dimension in `to`
///   and cannot be broadcasted (i.e., it is not 1 or does not match).
fn broadcast_shape(shape: &[usize], to: &[usize]) -> Vec<usize> {
    let to = to.to_vec();

    if to.len() < shape.len() {
        panic!("cannot broadcast {shape:?} to shape {to:?} with fewer dimensions")
    }

    let last_ndims = &to[to.len() - shape.len()..];

    for axis in 0..shape.len() {
        if shape[axis] != 1 && shape[axis] != last_ndims[axis] {
            panic!("broadcasting {shape:?} is not compatible with the desired shape {to:?}");
        }
    }

    to
}

/// Calculates the broadcasted strides for a tensor to match the specified broadcast shape.
///
/// This is done be left-padding the original stride with zeros until it matches the desired dimension.
/// The stride is set to 0 for any axes that have been repeated and kept the same otherwise.
///
/// # Panics
/// - If the number of dimensions in `broadcast_shape` is less than the number of dimensions in `original_shape`.
/// - If a dimension in `original_shape` does not equal the corresponding dimension in `broadcast_shape`
///   and cannot be broadcasted (i.e., it is not 1 or does not match).
///
/// # Examples
///
/// ```ignore
/// let stride = vec![4, 1];
/// let original_shape = vec![2, 3];
/// let broadcast_shape = vec![3, 2, 3];
///
/// let result = broadcast_stride(&stride, &broadcast_shape, &original_shape);
/// assert_eq!(result, vec![0, 4, 1]);
/// ```
fn broadcast_stride(stride: &[usize],
                    broadcast_shape: &[usize],
                    original_shape: &[usize]) -> Vec<usize> {
    let ndims = broadcast_shape.len();

    if ndims < original_shape.len() {
        panic!("cannot broadcast {original_shape:?} to shape {broadcast_shape:?} with fewer dimensions");
    }

    let mut broadcast_stride = Vec::with_capacity(ndims);
    let original_first_axis = ndims - original_shape.len();

    broadcast_stride.resize(original_first_axis, 0);  // new dimensions get a zero stride

    for axis in original_first_axis..ndims {
        let original_axis_length = original_shape[axis - original_first_axis];

        if original_axis_length == 1 {
            broadcast_stride.push(0);
        } else if original_axis_length == broadcast_shape[axis] {
            broadcast_stride.push(stride[axis - original_first_axis]);
        } else {
            panic!("broadcasting {original_shape:?} is not compatible with the desired shape {broadcast_shape:?}");
        }
    }

    broadcast_stride
}

/// Broadcasts two compatible shapes together and returns the resulting shape.
///
/// Broadcasting follows the rules of NumPy-style broadcasting:
/// - The smaller shape is left-padded with ones until it matches the length of the other shape
/// - If one of the shapes is of length 1 at a particular axis, it can broadcast to the length of the other shape at that axis.
/// - If both shapes have differing lengths at a certain axis and neither is 1, the two shapes are deemed incompatible for broadcasting.
///
/// For example, if `first` is `[8, 1, 6]` and `second` is `[7, 1]`, then `second` is left-padded
/// to become `[1, 7, 1]`. The middle axis of `first` is repeated to have dimension 7 and the
/// first and last axes of `second` are repeated to have dimensions 8 and 6 respectively.
/// The resulting shape is `[8, 7, 6]`.
///
/// # Panics
/// - If the two shapes are incompatible for broadcasting
///
/// # Examples
/// ```ignore
/// let shape1 = vec![8, 1, 6];
/// let shape2 = vec![7, 1];
/// let result = broadcast_shapes(&shape1, &shape2);
/// assert_eq!(result, vec![8, 7, 6]);
/// ```
pub(crate) fn broadcast_shapes(first: &[usize], second: &[usize]) -> Vec<usize> {
    let mut shape1;
    let mut shape2;

    // pad shapes with ones to match in length
    if first.len() > second.len() {
        shape1 = pad(second, 1, first.len());
        shape2 = first.to_vec();
    } else {
        shape1 = pad(first, 1, second.len());
        shape2 = second.to_vec();
    }

    for axis in 0..shape1.len() {
        // If one of the shapes is 1 at a particular axis,
        // it can be repeated to match the length of the other's shape at that axis   
        if shape1[axis] == 1 {
            shape1[axis] = shape2[axis];
        } else if shape2[axis] == 1 {
            shape2[axis] = shape1[axis];
        }

        // if neither shape is 1 along axis, and they don't match, the shapes cannot be broadcast
        else if shape1[axis] != shape2[axis] {
            panic!("broadcasting {first:?} is not compatible with the desired shape {second:?}");
        }
    }

    shape1
}

/// Determines the axes that are broadcasted when broadcasting from the `original_shape` 
/// to the `broadcast_shape`.
///
/// # Panics
/// - If `broadcast_shape` has fewer dimensions than `original_shape`.
///
/// # Example
///
/// ```ignore
/// let broadcast_shape = vec![4, 3, 2];
/// let original_shape = vec![3, 1];
/// let axes = get_broadcasted_axes(&broadcast_shape, &original_shape);
/// assert_eq!(axes, vec![0, 2]);
/// ```
///
/// In this example:
/// - Dimension `0` in the `broadcast_shape` (size `4`) is broadcasted because `original_shape` is missing
///   that dimension.
/// - Dimension `2` in the `broadcast_shape` (size `2`) is broadcasted because `original_shape[1]` is `1`.
pub(crate) fn get_broadcasted_axes(broadcast_shape: &[usize],
                                   original_shape: &[usize]) -> Vec<isize> {

    if broadcast_shape.len() < original_shape.len() {
        panic!("cannot broadcast {original_shape:?} to shape {broadcast_shape:?} with fewer dimensions");
    }
    
    let ndims_diff = broadcast_shape.len() - original_shape.len();
    let mut axes = Vec::new();

    for i in 0..broadcast_shape.len() {
        let to_dim = broadcast_shape[i];
        let from_dim = if i < ndims_diff { 1 } else { original_shape[i - ndims_diff] };

        if from_dim == 1 && to_dim > 1 || i < ndims_diff {
            axes.push(i as isize);
        }
    }

    axes
}

#[cfg(test)]
mod tests {
    use crate::broadcast::{broadcast_shapes, get_broadcasted_axes};

    #[test]
    fn test_broadcast_shapes() {
        let shape1 = vec![5, 1];
        let shape2 = vec![2, 1, 3];

        let correct = vec![2, 5, 3];
        let output = broadcast_shapes(&shape1, &shape2);

        assert_eq!(output, correct);
    }

    #[test]
    fn test_get_broadcasted_axes() {
        // grad_shape: [3, 3]
        // original_shape: [3, 1]
        // axes to sum: [1]
        assert_eq!(get_broadcasted_axes(&[3, 3], &[3, 1]), vec![1]);

        // grad_shape: [2, 3]
        // original_shape: [3]
        // axes to sum: [0]
        assert_eq!(get_broadcasted_axes(&[2, 3], &[3]), vec![0]);

        // grad_shape: [8, 7, 6]
        // original_shape: [7, 1]
        // axes to sum: [0, 2]
        assert_eq!(get_broadcasted_axes(&[8, 7, 6], &[7, 1]), vec![0, 2]);
        
        // grad_shape: [4, 5, 6]
        // original_shape: [1, 5, 1]
        // axes to sum: [0, 2]
        assert_eq!(get_broadcasted_axes(&[4, 5, 6], &[1, 5, 1]), vec![0, 2]);

        // grad_shape: [5, 6]
        // original_shape: [1, 6]
        // axes to sum: [0]
        assert_eq!(get_broadcasted_axes(&[5, 6], &[1, 6]), vec![0]);

        // grad_shape: [5, 6]
        // original_shape: [5, 1]
        // axes to sum: [1]
        assert_eq!(get_broadcasted_axes(&[5, 6], &[5, 1]), vec![1]);
    }
}
