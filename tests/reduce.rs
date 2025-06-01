use chela::*;
use num::NumCast;
use paste::paste;

#[test]
#[should_panic]
fn test_reduce_panic() {
    let tensor = NdArray::new([[1, 1], [2, 2], [3, 3]]);
    tensor.sum_along([0, 0]);
}


test_for_all_numeric_dtypes!(
    test_sum, {
        let tensor = NdArray::new([[1, 1], [2, 2], [3, 3]]).astype::<T>();

        let correct = NdArray::new([2, 4, 6]).astype::<T>();
        let output = tensor.sum_along(1);
        assert_eq!(output, correct);

        let output = tensor.sum_along(Axis(1));
        assert_eq!(output, correct);

        let correct = tensor.clone();
        let output = tensor.sum_along([]);
        assert_eq!(output, correct);

        let correct = NdArray::scalar(12).astype::<T>();
        let output = tensor.sum();
        assert_eq!(output, correct);
    }
);

test_for_common_numeric_dtypes!(
    test_sum2, {
        let two: T = NumCast::from(2).unwrap();

        for n in 1..23 {
            let tensor = NdArray::arange(0, n).astype::<T>();

            let correct = NdArray::scalar(n * n - n).astype::<T>();
            let output = tensor.sum() * two;
            assert_eq!(output, correct);
        }
    }
);

test_for_common_numeric_dtypes!(
    test_sum_slice, {
        let tensor = NdArray::new([
            [[1, 5, 3], [2, 9, 4]],
            [[2, 6, 4], [3, 8, 3]],
            [[3, 7, 5], [4, 7, 2]],
            [[4, 8, 6], [5, 6, 1]]
        ]).astype::<T>();

        let slice = tensor.slice(s![1..3, .., 0..=1]);

        let correct = NdArray::new([12, 28]).astype::<T>();
        let output = slice.sum_along([0, 1]);
        assert_eq!(output, correct);

        let correct = NdArray::scalar(40).astype::<T>();
        let output = slice.sum();
        assert_eq!(output, correct);

        // uniform stride but non-contiguous
        let slice = tensor.slice(s![.., .., 0]);

        let correct = NdArray::scalar(24).astype::<T>();
        let output = slice.sum();
        assert_eq!(output, correct);

        let correct = NdArray::new([3, 5, 7, 9]).astype::<T>();
        let output = slice.sum_along([1]);
        assert_eq!(output, correct);

        let correct = NdArray::new([10, 14]).astype::<T>();
        let output = slice.sum_along([0]);
        assert_eq!(output, correct);
    }
);

#[test]
fn test_reduce_multiply() {
    let tensor = NdArray::new([[1, 1], [2, 2], [3, 3]]);

    let correct = NdArray::new([1, 4, 9]);
    let output = tensor.product_along(1);
    assert_eq!(output, correct);

    let correct = tensor.clone();
    let output = tensor.product_along([]);
    assert_eq!(output, correct);

    let correct = NdArray::scalar(36);
    let output = tensor.product();
    assert_eq!(output, correct);
}

#[test]
fn test_reduce_mean() {
    let tensor = NdArray::new([[1f32, 3.0], [2.0, 4.0], [3.0, 5.0]]);

    let correct = NdArray::new([2.0f32, 3.0, 4.0]);
    let output = tensor.mean_along(1);
    assert_eq!(output, correct);

    let correct = NdArray::new([[1.0f32, 3.0], [2.0, 4.0], [3.0, 5.0]]);
    let output = tensor.mean_along([]);
    assert_eq!(output, correct);

    let correct = NdArray::scalar(3.0f32);
    let output = tensor.mean();
    assert_eq!(output, correct);
}

#[test]
fn test_reduce_min() {
    let tensor = NdArray::new([[1, 3], [2, 4], [3, 5]]);

    let correct = NdArray::new([1, 2, 3]);
    let output = tensor.min_along(1);
    assert_eq!(output, correct);

    let correct = NdArray::scalar(1);
    let output = tensor.min();
    assert_eq!(output, correct);
}

#[test]
fn test_reduce_max() {
    let tensor = NdArray::new([[1, 3], [2, 4], [3, 5]]);

    let correct = NdArray::new([3, 4, 5]);
    let output = tensor.max_along(1);
    assert_eq!(output, correct);

    let correct = NdArray::scalar(5);
    let output = tensor.max();
    assert_eq!(output, correct);
}

// ChatGPT generated
#[test]
fn test_tensor_operations() {
    let tensor = NdArray::new([[1i32, 3], [2, 4], [3, 5]]);

    // Sum tests
    assert_eq!(tensor.sum_along(1), NdArray::new([4, 6, 8]));
    assert_eq!(tensor.sum(), NdArray::scalar(18));

    // Product tests
    assert_eq!(tensor.product_along(1), NdArray::new([3, 8, 15]));
    assert_eq!(tensor.product(), NdArray::scalar(360));

    // Min & Max tests
    assert_eq!(tensor.min_along(1), NdArray::new([1, 2, 3]));
    assert_eq!(tensor.max_along(1), NdArray::new([3, 4, 5]));

    // Floating-point tests
    let tensor_f64 = NdArray::new([[1.0f64, 3.0], [2.0, 4.0], [3.0, 5.0]]);
    assert_eq!(tensor_f64.mean_along(1), NdArray::new([2.0, 3.0, 4.0]));
    assert_eq!(tensor_f64.sum_along(1), NdArray::new([4.0, 6.0, 8.0]));
    assert_eq!(tensor_f64.product_along(1), NdArray::new([3.0, 8.0, 15.0]));

    // Non-contiguous slices
    let slice = tensor.slice(s![.., 0]);
    assert_eq!(slice.sum(), NdArray::scalar(6));

    // Additional test cases
    let tensor_usize = NdArray::new([[1, 2], [3, 4], [5, 6]]);
    assert_eq!(tensor_usize.sum_along(1), NdArray::new([3, 7, 11]));
    assert_eq!(tensor_usize.product_along(1), NdArray::new([2, 12, 30]));

    let tensor_f32 = NdArray::new([[2.0f32, 4.0], [6.0, 8.0]]);
    assert_eq!(tensor_f32.mean_along(1), NdArray::new([3.0f32, 7.0]));
    assert_eq!(tensor_f32.product(), NdArray::scalar(384.0));

    let tensor_min_max = NdArray::new([[10i32, 20], [5, 15], [7, 9]]);
    assert_eq!(tensor_min_max.min_along(1), NdArray::new([10, 5, 7]));
    assert_eq!(tensor_min_max.max_along(1), NdArray::new([20, 15, 9]));

    let slice2 = tensor_min_max.slice(s![.., 1]);
    assert_eq!(slice2.sum(), NdArray::scalar(44));
}

#[test]
fn test_sum_operations() {
    let tensor = NdArray::new([[1i32, 3], [2, 4], [3, 5]]);

    // Sliced sum (column 0)
    let slice = tensor.slice(s![.., 0]);
    assert_eq!(slice.sum(), NdArray::scalar(6));

    // Sliced sum (row 1)
    let slice = tensor.slice(s![1, ..]);
    assert_eq!(slice.sum(), NdArray::scalar(6));

    // Sliced sum (first two rows, all columns)
    let slice = tensor.slice(s![0..2, ..]);
    assert_eq!(slice.sum(), NdArray::scalar(10));

    // Sliced sum (last two rows, column 1)
    let slice = tensor.slice(s![1.., 1]);
    assert_eq!(slice.sum(), NdArray::scalar(9));

    // Higher dimensional tensor (3D)
    let tensor_3d = NdArray::new([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]);
    assert_eq!(tensor_3d.sum_along(2), NdArray::new([[3, 7], [11, 15]]));
    assert_eq!(tensor_3d.sum_along(1), NdArray::new([[4, 6], [12, 14]]));
    assert_eq!(tensor_3d.sum(), NdArray::scalar(36));

    // Slicing along higher dimensions
    let slice = tensor_3d.slice(s![.., 0, ..]);
    assert_eq!(slice.sum(), NdArray::scalar(14));
    let slice = tensor_3d.slice(s![1, .., 0]);
    assert_eq!(slice.sum(), NdArray::scalar(12));
}

#[test]
fn test_product_operations() {
    let tensor = NdArray::new([[1i32, 3], [2, 4], [3, 5]]);

    // Sliced product (column 1)
    let slice = tensor.slice(s![.., 1]);
    assert_eq!(slice.product(), NdArray::scalar(60));

    // Sliced product (row 0)
    let slice = tensor.slice(s![0, ..]);
    assert_eq!(slice.product(), NdArray::scalar(3));

    // Sliced product (first two rows, all columns)
    let slice = tensor.slice(s![0..2, ..]);
    assert_eq!(slice.product(), NdArray::scalar(24));

    // Sliced product (last two rows, column 0)
    let slice = tensor.slice(s![1.., 0]);
    assert_eq!(slice.product(), NdArray::scalar(6));

    // Higher dimensional tensor (3D)
    let tensor_3d = NdArray::new([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]);
    assert_eq!(tensor_3d.product_along(2), NdArray::new([[2, 12], [30, 56]]));
    assert_eq!(tensor_3d.product_along(1), NdArray::new([[3, 8], [35, 48]]));
    assert_eq!(tensor_3d.product(), NdArray::scalar(40320));

    // Slicing along higher dimensions
    let slice = tensor_3d.slice(s![.., 0, ..]);
    assert_eq!(slice.product(), NdArray::scalar(60));
    let slice = tensor_3d.slice(s![1, .., 1]);
    assert_eq!(slice.product(), NdArray::scalar(48));
}

#[test]
fn test_reduce_min_f32() {
    let tensor = NdArray::new([[1.0f32, 3.0], [2.0, 4.0], [3.0, 5.0]]);

    let correct = NdArray::new([1.0f32, 2.0, 3.0]);
    let output = tensor.min_along(1);
    assert_eq!(output, correct);

    let correct = NdArray::scalar(1.0f32);
    let output = tensor.min();
    assert_eq!(output, correct);
}

#[test]
fn test_reduce_max_f64() {
    let tensor = NdArray::new([[1.0f64, 3.0], [2.0, 4.0], [3.0, 5.0]]);

    let correct = NdArray::new([3.0f64, 4.0, 5.0]);
    let output = tensor.max_along(1);
    assert_eq!(output, correct);

    let correct = NdArray::scalar(5.0f64);
    let output = tensor.max();
    assert_eq!(output, correct);
}

#[test]
fn test_reduce_min_slice_f64() {
    let tensor = NdArray::new([
        [[-1f64, 5.0, 36.0], [2.0, 9.0, -4.0]],
        [[12.0, 56.0, 47.0], [3.0, 8.0, -36.0]],
        [[23.0, -67.0, 5.0], [-4.0, 7.0, 2.0]],
        [[-40.0, 80.0, 62.0], [5.0, 6.0, -90.0]]
    ]);
    let slice = tensor.slice(s![1..3, .., 0..=1]);

    let correct = NdArray::new([-4f64, -67.0]);
    let output = slice.min_along([0, 1]);
    assert_eq!(output, correct);

    let correct = NdArray::scalar(-67.0);
    let output = slice.min();
    assert_eq!(output, correct);

    // uniform stride but non-contiguous
    let slice = tensor.slice(s![.., .., 0]);

    let correct = NdArray::scalar(-40.0);
    let output = slice.min();
    assert_eq!(output, correct);

    let correct = NdArray::new([-1f64, 3.0, -4.0, -40.0]);
    let output = slice.min_along([1]);
    assert_eq!(output, correct);

    let correct = NdArray::new([-40f64, -4.0]);
    let output = slice.min_along([0]);
    assert_eq!(output, correct);
}

#[test]
fn test_reduce_min_slice_f32() {
    let tensor = NdArray::new([
        [[-1f32, 5.0, 36.0], [2.0, 9.0, -4.0]],
        [[12.0, 56.0, 47.0], [3.0, 8.0, -36.0]],
        [[23.0, -67.0, 5.0], [-4.0, 7.0, 2.0]],
        [[-40.0, 80.0, 62.0], [5.0, 6.0, -90.0]]
    ]);
    let slice = tensor.slice(s![1..3, .., 0..=1]);

    let correct = NdArray::new([-4f32, -67.0]);
    let output = slice.min_along([0, 1]);
    assert_eq!(output, correct);

    let correct = NdArray::scalar(-67f32);
    let output = slice.min();
    assert_eq!(output, correct);

    // uniform stride but non-contiguous
    let slice = tensor.slice(s![.., .., 0]);

    let correct = NdArray::scalar(-40.0);
    let output = slice.min();
    assert_eq!(output, correct);

    let correct = NdArray::new([-1f32, 3.0, -4.0, -40.0]);
    let output = slice.min_along([1]);
    assert_eq!(output, correct);

    let correct = NdArray::new([-40f32, -4.0]);
    let output = slice.min_along([0]);
    assert_eq!(output, correct);
}

#[test]
fn test_reduce_min_slice_i32() {
    let tensor = NdArray::new([
        [[-1, 5, 36], [2, 9, -4]],
        [[12, 56, 47], [3, 8, -36]],
        [[23, -67, 5], [-4, 7, 2]],
        [[-40, 80, 62], [5, 6, -90]]
    ]);
    let slice = tensor.slice(s![1..3, .., 0..=1]);

    let correct = NdArray::new([-4, -67]);
    let output = slice.min_along([0, 1]);
    assert_eq!(output, correct);

    let correct = NdArray::scalar(-67);
    let output = slice.min();
    assert_eq!(output, correct);

    // uniform stride but non-contiguous
    let slice = tensor.slice(s![.., .., 0]);

    let correct = NdArray::scalar(-40);
    let output = slice.min();
    assert_eq!(output, correct);

    let correct = NdArray::new([-1, 3, -4, -40]);
    let output = slice.min_along([1]);
    assert_eq!(output, correct);

    let correct = NdArray::new([-40, -4]);
    let output = slice.min_along([0]);
    assert_eq!(output, correct);
}

#[test]
fn test_reduce_max_slice_f32() {
    let tensor = NdArray::new([
        [[-1f32, 5.0, 36.0], [2.0, 9.0, -4.0]],
        [[12.0, 56.0, 47.0], [3.0, 8.0, -36.0]],
        [[23.0, -67.0, 5.0], [-4.0, 7.0, 2.0]],
        [[-40.0, 80.0, 62.0], [5.0, 6.0, -90.0]]
    ]);
    let slice = tensor.slice(s![1..3, .., 0..=1]);

    let correct = NdArray::new([23f32, 56.0]);
    let output = slice.max_along([0, 1]);
    assert_eq!(output, correct);

    let correct = NdArray::scalar(56f32);
    let output = slice.max();
    assert_eq!(output, correct);

    // uniform stride but non-contiguous
    let slice = tensor.slice(s![.., .., 0]);

    let correct = NdArray::scalar(23.0);
    let output = slice.max();
    assert_eq!(output, correct);

    let correct = NdArray::new([2f32, 12.0, 23.0, 5.0]);
    let output = slice.max_along([1]);
    assert_eq!(output, correct);

    let correct = NdArray::new([23f32, 5.0]);
    let output = slice.max_along([0]);
    assert_eq!(output, correct);
}

#[test]
fn test_reduce_max_slice_f64() {
    let tensor = NdArray::new([
        [[-1f64, 5.0, 36.0], [2.0, 9.0, -4.0]],
        [[12.0, 56.0, 47.0], [3.0, 8.0, -36.0]],
        [[23.0, -67.0, 5.0], [-4.0, 7.0, 2.0]],
        [[-40.0, 80.0, 62.0], [5.0, 6.0, -90.0]]
    ]);
    let slice = tensor.slice(s![1..3, .., 0..=1]);

    let correct = NdArray::new([23f64, 56.0]);
    let output = slice.max_along([0, 1]);
    assert_eq!(output, correct);

    let correct = NdArray::scalar(56f64);
    let output = slice.max();
    assert_eq!(output, correct);

    // uniform stride but non-contiguous
    let slice = tensor.slice(s![.., .., 0]);

    let correct = NdArray::scalar(23.0);
    let output = slice.max();
    assert_eq!(output, correct);

    let correct = NdArray::new([2f64, 12.0, 23.0, 5.0]);
    let output = slice.max_along([1]);
    assert_eq!(output, correct);

    let correct = NdArray::new([23f64, 5.0]);
    let output = slice.max_along([0]);
    assert_eq!(output, correct);
}

#[test]
fn test_reduce_max_slice_i32() {
    let tensor = NdArray::new([
        [[-1, 5, 36], [2, 9, -4]],
        [[12, 56, 47], [3, 8, -36]],
        [[23, -67, 5], [-4, 7, 2]],
        [[-40, 80, 62], [5, 6, -90]]
    ]);
    let slice = tensor.slice(s![1..3, .., 0..=1]);

    let correct = NdArray::new([23, 56]);
    let output = slice.max_along([0, 1]);
    assert_eq!(output, correct);

    let correct = NdArray::scalar(56);
    let output = slice.max();
    assert_eq!(output, correct);

    // uniform stride but non-contiguous
    let slice = tensor.slice(s![.., .., 0]);

    let correct = NdArray::scalar(23);
    let output = slice.max();
    assert_eq!(output, correct);

    let correct = NdArray::new([2, 12, 23, 5]);
    let output = slice.max_along([1]);
    assert_eq!(output, correct);

    let correct = NdArray::new([23, 5]);
    let output = slice.max_along([0]);
    assert_eq!(output, correct);
}

#[test]
fn test_reduce_max_magnitude_f32() {
    let tensor = NdArray::new([
        [[-3f32, 5.0, 36.0], [2.0, 9.0, -4.0]],
        [[12.0, 56.0, 47.0], [3.0, 8.0, -36.0]],
        [[23.0, -67.0, 5.0], [-4.0, 7.0, 2.0]],
        [[-40.0, -80.0, 62.0], [-45.0, 6.0, -90.0]]
    ]);
    let slice = tensor.slice(s![1..3, .., 0..=1]);

    let correct = NdArray::new([23f32, 67.0]);
    let output = slice.max_magnitude_along([0, 1]);
    assert_eq!(output, correct);

    let correct = NdArray::scalar(67f32);
    let output = slice.max_magnitude();
    assert_eq!(output, correct);

    // uniform stride but non-contiguous
    let slice = tensor.slice(s![.., .., 0]);

    let correct = NdArray::scalar(45.0);
    let output = slice.max_magnitude();
    assert_eq!(output, correct);

    let correct = NdArray::new([3f32, 12.0, 23.0, 45.0]);
    let output = slice.max_magnitude_along([1]);
    assert_eq!(output, correct);

    let correct = NdArray::new([40f32, 45.0]);
    let output = slice.max_magnitude_along([0]);
    assert_eq!(output, correct);
}

#[test]
fn test_reduce_max_magnitude_f64() {
    let tensor = NdArray::new([
        [[-3f64, 5.0, 36.0], [2.0, 9.0, -4.0]],
        [[12.0, 56.0, 47.0], [3.0, 8.0, -36.0]],
        [[23.0, -67.0, 5.0], [-4.0, 7.0, 2.0]],
        [[-40.0, -80.0, 62.0], [-45.0, 6.0, -90.0]]
    ]);
    let slice = tensor.slice(s![1..3, .., 0..=1]);

    let correct = NdArray::new([23f64, 67.0]);
    let output = slice.max_magnitude_along([0, 1]);
    assert_eq!(output, correct);

    let correct = NdArray::scalar(67f64);
    let output = slice.max_magnitude();
    assert_eq!(output, correct);

    // uniform stride but non-contiguous
    let slice = tensor.slice(s![.., .., 0]);

    let correct = NdArray::scalar(45.0);
    let output = slice.max_magnitude();
    assert_eq!(output, correct);

    let correct = NdArray::new([3f64, 12.0, 23.0, 45.0]);
    let output = slice.max_magnitude_along([1]);
    assert_eq!(output, correct);

    let correct = NdArray::new([40f64, 45.0]);
    let output = slice.max_magnitude_along([0]);
    assert_eq!(output, correct);
}

#[test]
fn test_reduce_max_magnitude_i32() {
    let tensor = NdArray::new([
        [[-3, 5, 36], [2, 9, -4]],
        [[12, 56, 47], [3, 8, -36]],
        [[23, -67, 5], [-4, 7, 2]],
        [[-40, -80, 62], [-45, 6, -90]]
    ]);
    let slice = tensor.slice(s![1..3, .., 0..=1]);

    let correct = NdArray::new([23, 67]);
    let output = slice.max_magnitude_along([0, 1]);
    assert_eq!(output, correct);

    let correct = NdArray::scalar(67);
    let output = slice.max_magnitude();
    assert_eq!(output, correct);

    // uniform stride but non-contiguous
    let slice = tensor.slice(s![.., .., 0]);

    let correct = NdArray::scalar(45);
    let output = slice.max_magnitude();
    assert_eq!(output, correct);

    let correct = NdArray::new([3, 12, 23, 45]);
    let output = slice.max_magnitude_along([1]);
    assert_eq!(output, correct);

    let correct = NdArray::new([40, 45]);
    let output = slice.max_magnitude_along([0]);
    assert_eq!(output, correct);
}
