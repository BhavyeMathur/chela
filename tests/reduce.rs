use chela::*;

#[test]
#[should_panic]
fn test_reduce_panic() {
    let tensor = Tensor::from([[1, 1], [2, 2], [3, 3]]);
    tensor.sum_along([0, 0]);
}

#[test]
fn test_reduce_sum_f32() {
    let tensor = Tensor::from([[1f32, 1.0], [2.0, 2.0], [3.0, 3.0]]);

    let correct = Tensor::from([2f32, 4.0, 6.0]);
    let output = tensor.sum_along(1);
    assert_eq!(output, correct);

    let output = tensor.sum_along(Axis(1));
    assert_eq!(output, correct);

    let correct = tensor.clone();
    let output = tensor.sum_along([]);
    assert_eq!(output, correct);

    let correct = Tensor::scalar(12.0);
    let output = tensor.sum();
    assert_eq!(output, correct);
}

#[test]
fn test_reduce_sum_slice_u128() {
    let tensor = Tensor::from([
        [[1u128, 5, 3], [2, 9, 4]],
        [[2, 6, 4], [3, 8, 3]],
        [[3, 7, 5], [4, 7, 2]],
        [[4, 8, 6], [5, 6, 1]]
    ]);
    let slice = tensor.slice(s![1..3, .., 0..=1]);

    let correct = Tensor::from([12u128, 28]);
    let output = slice.sum_along([0, 1]);
    assert_eq!(output, correct);

    let correct = Tensor::scalar(40u128);
    let output = slice.sum();
    assert_eq!(output, correct);

    // uniform stride but non-contiguous
    let slice = tensor.slice(s![.., .., 0]);

    let correct = Tensor::scalar(24u128);
    let output = slice.sum();
    assert_eq!(output, correct);

    let correct = Tensor::from([3u128, 5, 7, 9]);
    let output = slice.sum_along([1]);
    assert_eq!(output, correct);

    let correct = Tensor::from([10u128, 14]);
    let output = slice.sum_along([0]);
    assert_eq!(output, correct);
}

#[test]
fn test_reduce_sum_slice_f32() {
    let tensor = Tensor::from([
        [[1f32, 5.0, 3.0], [2.0, 9.0, 4.0]],
        [[2.0, 6.0, 4.0], [3.0, 8.0, 3.0]],
        [[3.0, 7.0, 5.0], [4.0, 7.0, 2.0]],
        [[4.0, 8.0, 6.0], [5.0, 6.0, 1.0]]
    ]);
    // non-uniform stride, non-contiguous
    let slice = tensor.slice(s![1..3, .., 0..=1]);

    let correct = Tensor::from([12f32, 28.0]);
    let output = slice.sum_along([0, 1]);
    assert_eq!(output, correct);

    let correct = Tensor::scalar(40.0);
    let output = slice.sum();
    assert_eq!(output, correct);

    // uniform stride but non-contiguous
    let slice = tensor.slice(s![.., .., 0]);

    let correct = Tensor::scalar(24.0);
    let output = slice.sum();
    assert_eq!(output, correct);

    let correct = Tensor::from([3f32, 5.0, 7.0, 9.0]);
    let output = slice.sum_along([1]);
    assert_eq!(output, correct);

    let correct = Tensor::from([10f32, 14.0]);
    let output = slice.sum_along([0]);
    assert_eq!(output, correct);
}

#[test]
fn test_reduce_sum_slice_f64() {
    let tensor = Tensor::from([
        [[1f64, 5.0, 3.0], [2.0, 9.0, 4.0]],
        [[2.0, 6.0, 4.0], [3.0, 8.0, 3.0]],
        [[3.0, 7.0, 5.0], [4.0, 7.0, 2.0]],
        [[4.0, 8.0, 6.0], [5.0, 6.0, 1.0]]
    ]);
    let slice = tensor.slice(s![1..3, .., 0..=1]);
    assert!(!slice.is_contiguous());
    assert!(slice.has_uniform_stride().is_none());

    let correct = Tensor::from([12f64, 28.0]);
    let output = slice.sum_along([0, 1]);
    assert_eq!(output, correct);

    let correct = Tensor::scalar(40.0);
    let output = slice.sum();
    assert_eq!(output, correct);

    // uniform stride but non-contiguous
    let slice = tensor.slice(s![.., .., 0]);
    assert!(!slice.is_contiguous());
    assert!(slice.has_uniform_stride().is_some());

    let correct = Tensor::scalar(24.0);
    let output = slice.sum();
    assert_eq!(output, correct);

    let correct = Tensor::from([3f64, 5.0, 7.0, 9.0]);
    let output = slice.sum_along([1]);
    assert_eq!(output, correct);

    let correct = Tensor::from([10f64, 14.0]);
    let output = slice.sum_along([0]);
    assert_eq!(output, correct);
}

#[test]
fn test_reduce_sum_f64() {
    let tensor = Tensor::from([[1f64, 1.0], [2.0, 2.0], [3.0, 3.0]]);

    let correct = Tensor::from([2f64, 4.0, 6.0]);
    let output = tensor.sum_along(1);
    assert_eq!(output, correct);

    let output = tensor.sum_along(Axis(1));
    assert_eq!(output, correct);

    let correct = tensor.clone();
    let output = tensor.sum_along([]);
    assert_eq!(output, correct);

    let correct = Tensor::scalar(12.0);
    let output = tensor.sum();
    assert_eq!(output, correct);
}

#[test]
fn test_reduce_sum_i32() {
    let tensor = Tensor::from([[1i32, 1], [2, 2], [3, 3]]);

    let correct = Tensor::from([2, 4, 6]);
    let output = tensor.sum_along(1);
    assert_eq!(output, correct);

    let output = tensor.sum_along(Axis(1));
    assert_eq!(output, correct);

    let correct = tensor.clone();
    let output = tensor.sum_along([]);
    assert_eq!(output, correct);

    let correct = Tensor::scalar(12);
    let output = tensor.sum();
    assert_eq!(output, correct);
}

#[test]
fn test_reduce_multiply() {
    let tensor = Tensor::from([[1, 1], [2, 2], [3, 3]]);

    let correct = Tensor::from([1, 4, 9]);
    let output = tensor.product_along(1);
    assert_eq!(output, correct);

    let correct = tensor.clone();
    let output = tensor.product_along([]);
    assert_eq!(output, correct);

    let correct = Tensor::scalar(36);
    let output = tensor.product();
    assert_eq!(output, correct);
}

#[test]
fn test_reduce_mean() {
    let tensor = Tensor::from([[1f32, 3.0], [2.0, 4.0], [3.0, 5.0]]);

    let correct = Tensor::from([2.0f32, 3.0, 4.0]);
    let output = tensor.mean_along(1);
    assert_eq!(output, correct);

    let correct = Tensor::from([[1.0f32, 3.0], [2.0, 4.0], [3.0, 5.0]]);
    let output = tensor.mean_along([]);
    assert_eq!(output, correct);

    let correct = Tensor::scalar(3.0f32);
    let output = tensor.mean();
    assert_eq!(output, correct);
}

#[test]
fn test_reduce_min() {
    let tensor = Tensor::from([[1, 3], [2, 4], [3, 5]]);

    let correct = Tensor::from([1, 2, 3]);
    let output = tensor.min_along(1);
    assert_eq!(output, correct);

    let correct = Tensor::scalar(1);
    let output = tensor.min();
    assert_eq!(output, correct);
}

#[test]
fn test_reduce_max() {
    let tensor = Tensor::from([[1, 3], [2, 4], [3, 5]]);

    let correct = Tensor::from([3, 4, 5]);
    let output = tensor.max_along(1);
    assert_eq!(output, correct);

    let correct = Tensor::scalar(5);
    let output = tensor.max();
    assert_eq!(output, correct);
}

// ChatGPT generated
#[test]
fn test_tensor_operations() {
    let tensor = Tensor::from([[1i32, 3], [2, 4], [3, 5]]);

    // Sum tests
    assert_eq!(tensor.sum_along(1), Tensor::from([4, 6, 8]));
    assert_eq!(tensor.sum(), Tensor::scalar(18));

    // Product tests
    assert_eq!(tensor.product_along(1), Tensor::from([3, 8, 15]));
    assert_eq!(tensor.product(), Tensor::scalar(360));

    // Min & Max tests
    assert_eq!(tensor.min_along(1), Tensor::from([1, 2, 3]));
    assert_eq!(tensor.max_along(1), Tensor::from([3, 4, 5]));

    // Floating-point tests
    let tensor_f64 = Tensor::from([[1.0f64, 3.0], [2.0, 4.0], [3.0, 5.0]]);
    assert_eq!(tensor_f64.mean_along(1), Tensor::from([2.0, 3.0, 4.0]));
    assert_eq!(tensor_f64.sum_along(1), Tensor::from([4.0, 6.0, 8.0]));
    assert_eq!(tensor_f64.product_along(1), Tensor::from([3.0, 8.0, 15.0]));

    // Non-contiguous slices
    let slice = tensor.slice(s![.., 0]);
    assert_eq!(slice.sum(), Tensor::scalar(6));

    // Additional test cases
    let tensor_usize = Tensor::from([[1, 2], [3, 4], [5, 6]]);
    assert_eq!(tensor_usize.sum_along(1), Tensor::from([3, 7, 11]));
    assert_eq!(tensor_usize.product_along(1), Tensor::from([2, 12, 30]));

    let tensor_f32 = Tensor::from([[2.0f32, 4.0], [6.0, 8.0]]);
    assert_eq!(tensor_f32.mean_along(1), Tensor::from([3.0f32, 7.0]));
    assert_eq!(tensor_f32.product(), Tensor::scalar(384.0));

    let tensor_min_max = Tensor::from([[10i32, 20], [5, 15], [7, 9]]);
    assert_eq!(tensor_min_max.min_along(1), Tensor::from([10, 5, 7]));
    assert_eq!(tensor_min_max.max_along(1), Tensor::from([20, 15, 9]));

    let slice2 = tensor_min_max.slice(s![.., 1]);
    assert_eq!(slice2.sum(), Tensor::scalar(44));
}

#[test]
fn test_sum_operations() {
    let tensor = Tensor::from([[1i32, 3], [2, 4], [3, 5]]);

    // Sliced sum (column 0)
    let slice = tensor.slice(s![.., 0]);
    assert_eq!(slice.sum(), Tensor::scalar(6));

    // Sliced sum (row 1)
    let slice = tensor.slice(s![1, ..]);
    assert_eq!(slice.sum(), Tensor::scalar(6));

    // Sliced sum (first two rows, all columns)
    let slice = tensor.slice(s![0..2, ..]);
    assert_eq!(slice.sum(), Tensor::scalar(10));

    // Sliced sum (last two rows, column 1)
    let slice = tensor.slice(s![1.., 1]);
    assert_eq!(slice.sum(), Tensor::scalar(9));

    // Higher dimensional tensor (3D)
    let tensor_3d = Tensor::from([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]);
    assert_eq!(tensor_3d.sum_along(2), Tensor::from([[3, 7], [11, 15]]));
    assert_eq!(tensor_3d.sum_along(1), Tensor::from([[4, 6], [12, 14]]));
    assert_eq!(tensor_3d.sum(), Tensor::scalar(36));

    // Slicing along higher dimensions
    let slice = tensor_3d.slice(s![.., 0, ..]);
    assert_eq!(slice.sum(), Tensor::scalar(14));
    let slice = tensor_3d.slice(s![1, .., 0]);
    assert_eq!(slice.sum(), Tensor::scalar(12));
}

#[test]
fn test_product_operations() {
    let tensor = Tensor::from([[1i32, 3], [2, 4], [3, 5]]);

    // Sliced product (column 1)
    let slice = tensor.slice(s![.., 1]);
    assert_eq!(slice.product(), Tensor::scalar(60));

    // Sliced product (row 0)
    let slice = tensor.slice(s![0, ..]);
    assert_eq!(slice.product(), Tensor::scalar(3));

    // Sliced product (first two rows, all columns)
    let slice = tensor.slice(s![0..2, ..]);
    assert_eq!(slice.product(), Tensor::scalar(24));

    // Sliced product (last two rows, column 0)
    let slice = tensor.slice(s![1.., 0]);
    assert_eq!(slice.product(), Tensor::scalar(6));

    // Higher dimensional tensor (3D)
    let tensor_3d = Tensor::from([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]);
    assert_eq!(tensor_3d.product_along(2), Tensor::from([[2, 12], [30, 56]]));
    assert_eq!(tensor_3d.product_along(1), Tensor::from([[3, 8], [35, 48]]));
    assert_eq!(tensor_3d.product(), Tensor::scalar(40320));

    // Slicing along higher dimensions
    let slice = tensor_3d.slice(s![.., 0, ..]);
    assert_eq!(slice.product(), Tensor::scalar(60));
    let slice = tensor_3d.slice(s![1, .., 1]);
    assert_eq!(slice.product(), Tensor::scalar(48));
}

// TODO
// #[test]
// fn test_reduce_min_f32() {
//     let tensor = Tensor::from([[1.0f32, 3.0], [2.0, 4.0], [3.0, 5.0]]);
//
//     let correct = Tensor::from([1.0f32, 2.0, 3.0]);
//     let output = tensor.min_along(1);
//     assert_eq!(output, correct);
//
//     let correct = Tensor::scalar(1.0f32);
//     let output = tensor.min();
//     assert_eq!(output, correct);
// }
//
// #[test]
// fn test_reduce_max_f64() {
//     let tensor = Tensor::from([[1.0f64, 3.0], [2.0, 4.0], [3.0, 5.0]]);
//
//     let correct = Tensor::from([3.0f64, 4.0, 5.0]);
//     let output = tensor.max_along(1);
//     assert_eq!(output, correct);
//
//     let correct = Tensor::scalar(5.0f64);
//     let output = tensor.max();
//     assert_eq!(output, correct);
// }
