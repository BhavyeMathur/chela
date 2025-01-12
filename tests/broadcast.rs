use chela::*;

#[test]
fn test_broadcast() {
    let tensor = Tensor::from([1, 2, 3]);
    let tensor = tensor.broadcast_to([3, 3]);
    assert_eq!(tensor.shape(), [3, 3]);
    assert_eq!(tensor, Tensor::from([[1, 2, 3], [1, 2, 3], [1, 2, 3]]));
}

#[test]
fn test_broadcast_scalar_to_higher_dims() {
    let tensor = Tensor::from([42]);
    let tensor = tensor.broadcast_to([2, 3]);
    assert_eq!(tensor.shape(), [2, 3]);
    assert_eq!(tensor, Tensor::from([[42, 42, 42], [42, 42, 42]]));
}

#[test]
fn test_broadcast_matrix_to_higher_dims() {
    let tensor = Tensor::from([[1, 2], [3, 4]]);
    let tensor = tensor.broadcast_to([3, 2, 2]);
    assert_eq!(tensor.shape(), [3, 2, 2]);
    assert_eq!(tensor, Tensor::from([[[1, 2], [3, 4]], [[1, 2], [3, 4]], [[1, 2], [3, 4]]]));
}

#[test]
#[should_panic]
fn test_broadcast_incompatible_shapes() {
    let tensor = Tensor::from([1, 2, 3]);
    tensor.broadcast_to([3, 5]);
}

#[test]
fn test_broadcast_identity() {
    let tensor = Tensor::from([1, 2, 3]);
    let tensor = tensor.broadcast_to([3]);
    assert_eq!(tensor.shape(), [3]);
    assert_eq!(tensor, Tensor::from([1, 2, 3]));
}

#[test]
fn test_broadcast_unchanged() {
    let tensor = Tensor::from([1, 2, 3]);
    let tensor = tensor.broadcast_to([1, 3]);
    assert_eq!(tensor.shape(), [1, 3]);
    assert_eq!(tensor, Tensor::from([[1, 2, 3]]));
}

#[test]
fn test_broadcast_high_dims() {
    let tensor = Tensor::from([[1], [2], [3]]);
    let tensor = tensor.broadcast_to([3, 3, 3]);
    assert_eq!(tensor.shape(), [3, 3, 3]);
    assert_eq!(tensor, Tensor::from(
        [[[1, 1, 1], [2, 2, 2], [3, 3, 3]],
            [[1, 1, 1], [2, 2, 2], [3, 3, 3]],
            [[1, 1, 1], [2, 2, 2], [3, 3, 3]]]
    )
    );
}

#[test]
fn test_broadcast_single_dimensional_expansion() {
    let tensor = Tensor::from([1, 2, 3]);
    let tensor = tensor.broadcast_to([3, 1, 3]);
    assert_eq!(tensor.shape(), [3, 1, 3]);
    assert_eq!(tensor, Tensor::from([[[1, 2, 3]], [[1, 2, 3]], [[1, 2, 3]]]));
}