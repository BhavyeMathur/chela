use redstone_ml::*;
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

        // non-uniform stride and non-contiguous
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

test_for_all_numeric_dtypes!(
    test_reduce_multiply, {
        let tensor = NdArray::new([[1, 1], [2, 2], [3, 3]]).astype::<T>();

        let correct = NdArray::new([1, 4, 9]).astype::<T>();
        let output = tensor.product_along(1);
        assert_eq!(output, correct);

        let correct = tensor.clone();
        let output = tensor.product_along([]);
        assert_eq!(output, correct);

        let correct = NdArray::scalar(36).astype::<T>();
        let output = tensor.product();
        assert_eq!(output, correct);
    }
);

test_for_common_numeric_dtypes!(
    test_reduce_multiply_big, {
        // 14 elements
        let tensor = NdArray::new([1, 2, 3, 2, 4, 5, 1, 2, 3, 4, 5, 6, 7, 9]).astype::<T>();
        let output = tensor.product();
        assert_eq!(output, NdArray::scalar(10886400).astype::<T>());

        // 17 elements
        let tensor = NdArray::new([1, 2, 3, 2, 4, 5, 1, 2, 3, 4, 5, 6, 7, 9, 2, 3, 2]).astype::<T>();
        let output = tensor.product();
        assert_eq!(output, NdArray::scalar(130636800).astype::<T>());

        // 20 elements
        let tensor = NdArray::new([1, 2, 3, 2, 4, 5, 1, 2, 3, 4, 5, 6, 2, 9, 2, 3, 2, 6, 2, 3]).astype::<T>();
        let output = tensor.product();
        assert_eq!(output, NdArray::scalar(1343692800).astype::<T>());

        // 23 elements
        let tensor = NdArray::new([1, 2, 3, 2, 4, 5, 1, 2, 3, 4, 1, 5, 1, 2, 2, 3, 2, 6, 2, 3, 3, 6, 4]).astype::<T>();
        let output = tensor.product();
        assert_eq!(output, NdArray::scalar(1791590400).astype::<T>());
    }
);

test_for_common_numeric_dtypes!(
    test_product_slice, {
        let tensor = NdArray::new([
            [[1, 5, 3], [2, 9, 4]],
            [[2, 6, 4], [3, 8, 3]],
            [[3, 7, 5], [4, 7, 2]],
            [[4, 8, 6], [5, 6, 1]],
            [[1, 2, 3], [4, 5, 6]]
        ]).astype::<T>();

        // non-uniform stride and non-contiguous
        let slice = tensor.slice(s![1..3, .., 0..=1]);

        let correct = NdArray::new([72, 2352]).astype::<T>();
        let output = slice.product_along([0, 1]);
        assert_eq!(output, correct);

        let correct = NdArray::scalar(169344).astype::<T>();
        let output = slice.product();
        assert_eq!(output, correct);

        // uniform stride but non-contiguous
        let slice = tensor.slice(s![.., .., 0]);

        let correct = NdArray::scalar(11520).astype::<T>();
        let output = slice.product();
        assert_eq!(output, correct);

        let correct = NdArray::new([2, 6, 12, 20, 4]).astype::<T>();
        let output = slice.product_along([1]);
        assert_eq!(output, correct);

        let correct = NdArray::new([24, 480]).astype::<T>();
        let output = slice.product_along([0]);
        assert_eq!(output, correct);
    }
);

test_for_all_numeric_dtypes!(
    test_reduce_min, {
        let tensor = NdArray::new([[1, 3], [2, 4], [3, 5]]).astype::<T>();

        let correct = NdArray::new([1, 2, 3]).astype::<T>();
        let output = tensor.min_along(1);
        assert_eq!(output, correct);

        let correct = NdArray::scalar(1).astype::<T>();
        let output = tensor.min();
        assert_eq!(output, correct);
    }
);

test_for_common_numeric_dtypes!(
    test_reduce_min_big, {
        let tensor = NdArray::new([12, 23, 3, 22, 4, 5, 1, 26, 3, 4, 51, 62, 0, 9, 1]).astype::<T>();
        let output = tensor.min();
        assert_eq!(output, NdArray::scalar(0).astype::<T>());

        let tensor = NdArray::new([12, 2, 34, 28, 42, 54, 11, 2, 43, 4, 512, 6, 79, 91, 2, 3, 22]).astype::<T>();
        let output = tensor.min();
        assert_eq!(output, NdArray::scalar(2).astype::<T>());

        let tensor = NdArray::new([1, 2, 3, 2, 4, 5, 1, 2, 3, 4, 5, 6, 2, 9, 2, 3, 2, 6, 2, 3]).astype::<T>();
        let output = tensor.min();
        assert_eq!(output, NdArray::scalar(1).astype::<T>());

        let tensor = NdArray::new([1, 2, 3, 2, 4, 5, 1, 2, 3, 4, 1, 5, 0, 2, 2, 3, 2, 6, 2, 3, 3, 6, 4]).astype::<T>();
        let output = tensor.min();
        assert_eq!(output, NdArray::scalar(0).astype::<T>());
    }
);

test_for_all_numeric_dtypes!(
    test_reduce_max, {
        let tensor = NdArray::new([[1, 3], [2, 4], [3, 5]]).astype::<T>();

        let correct = NdArray::new([3, 4, 5]).astype::<T>();
        let output = tensor.max_along(1);
        assert_eq!(output, correct);

        let correct = NdArray::scalar(5).astype::<T>();
        let output = tensor.max();
        assert_eq!(output, correct);
    }
);

test_for_common_numeric_dtypes!(
    test_reduce_max_big, {
        let tensor = NdArray::new([12, 23, 3, 22, 4, 5, 1, 26, 3, 4, 51, 62, 0, 9, 1]).astype::<T>();
        let output = tensor.max();
        assert_eq!(output, NdArray::scalar(62).astype::<T>());

        let tensor = NdArray::new([12, 2, 34, 28, 42, 54, 11, 2, 43, 4, 512, 6, 79, 91, 2, 3, 22]).astype::<T>();
        let output = tensor.max();
        assert_eq!(output, NdArray::scalar(512).astype::<T>());

        let tensor = NdArray::new([1, 2, 3, 2, 4, 5, 1, 2, 3, 4, 52, 6, 2, 9, 2, 3, 2, 6, 2, 3]).astype::<T>();
        let output = tensor.max();
        assert_eq!(output, NdArray::scalar(52).astype::<T>());

        let tensor = NdArray::new([1, 2, 3, 2, 4, 5, 1, 2, 3, 4, 1, 5, 0, 200, 2, 3, 2, 6, 2, 3, 3, 6, 4]).astype::<T>();
        let output = tensor.max();
        assert_eq!(output, NdArray::scalar(200).astype::<T>());
    }
);

test_for_signed_dtypes!(
    test_min_slice, {
        let tensor = NdArray::new([
            [[-1, 5, 36], [2, 9, -4]],
            [[12, 56, 47], [3, 8, -36]],
            [[23, -67, 5], [-4, 7, 2]],
            [[-40, 80, 62], [5, 6, -90]],
            [[-41, 8, 62], [50, 6, -92]]
        ]).astype::<T>();

        //  non-uniform stride and non-contiguous
        let slice = tensor.slice(s![1..3, .., 0..=1]);

        let correct = NdArray::new([-4, -67]).astype::<T>();
        let output = slice.min_along([0, 1]);
        assert_eq!(output, correct);

        let correct = NdArray::scalar(-67).astype::<T>();
        let output = slice.min();
        assert_eq!(output, correct);

        // uniform stride but non-contiguous
        let slice = tensor.slice(s![.., .., 0]);

        let correct = NdArray::scalar(-41).astype::<T>();
        let output = slice.min();
        assert_eq!(output, correct);

        let correct = NdArray::new([-1, 3, -4, -40, -41]).astype::<T>();
        let output = slice.min_along([1]);
        assert_eq!(output, correct);

        let correct = NdArray::new([-41, -4]).astype::<T>();
        let output = slice.min_along([0]);
        assert_eq!(output, correct);
    }
);

test_for_signed_dtypes!(
    test_max_slice, {
        let tensor = NdArray::new([
            [[-1, 5, 36], [2, 9, -4]],
            [[12, 56, 47], [3, 8, -36]],
            [[23, -67, 5], [-4, 7, 2]],
            [[-40, 80, 62], [5, 6, -90]],
            [[-41, 8, 62], [50, 6, -92]]
        ]).astype::<T>();

        //  non-uniform stride and non-contiguous
        let slice = tensor.slice(s![1..3, .., 0..=1]).astype::<T>();

        let correct = NdArray::new([23, 56]).astype::<T>();
        let output = slice.max_along([0, 1]);
        assert_eq!(output, correct);

        let correct = NdArray::scalar(56).astype::<T>();
        let output = slice.max();
        assert_eq!(output, correct);

        // uniform stride but non-contiguous
        let slice = tensor.slice(s![.., .., 0]);

        let correct = NdArray::scalar(50).astype::<T>();
        let output = slice.max();
        assert_eq!(output, correct);

        let correct = NdArray::new([2, 12, 23, 5, 50]).astype::<T>();
        let output = slice.max_along([1]);
        assert_eq!(output, correct);

        let correct = NdArray::new([23, 50]).astype::<T>();
        let output = slice.max_along([0]);
        assert_eq!(output, correct);
    }
);

test_for_unsigned_dtypes!(
    test_min_magnitude, {
        let tensor = NdArray::new([
            [[1, 5, 36], [2, 9, 4]],
            [[12, 56, 47], [3, 8, 36]],
            [[23, 67, 5], [4, 7, 2]],
            [[40, 80, 62], [45, 6, 90]]
        ]).astype::<T>();

        // non-uniform stride and non-contiguous
        let slice = tensor.slice(s![1..3, .., 0..=1]);

        let correct = slice.min_along([0, 1]);
        let output = slice.min_magnitude_along([0, 1]);
        assert_eq!(output, correct);

        let correct = slice.min();
        let output = slice.min_magnitude();
        assert_eq!(output, correct);

        // uniform stride but non-contiguous
        let slice = tensor.slice(s![.., .., 0]);

        let correct = slice.min();
        let output = slice.min_magnitude();
        assert_eq!(output, correct);

        let correct = slice.min_along([1]);
        let output = slice.min_magnitude_along([1]);
        assert_eq!(output, correct);

        let correct = slice.min_along([0]);
        let output = slice.min_magnitude_along([0]);
        assert_eq!(output, correct);
    }
);

test_for_signed_dtypes!(
    test_min_magnitude, {
        let tensor = NdArray::new([
            [[-1, 5, 36], [2, 9, -4]],
            [[12, 56, 47], [-3, 8, -36]],
            [[23, -67, 5], [-4, 7, 2]],
            [[-40, -80, 62], [-45, 6, -90]]
        ]).astype::<T>();

        // contiguous
        let correct = NdArray::scalar(1).astype::<T>();
        let output = tensor.min_magnitude();
        assert_eq!(output, correct);

        // non-uniform stride and non-contiguous
        let slice = tensor.slice(s![1..3, .., 0..=1]);

        let correct = NdArray::new([3, 7]).astype::<T>();
        let output = slice.min_magnitude_along([0, 1]);
        assert_eq!(output, correct);

        let correct = NdArray::scalar(3).astype::<T>();
        let output = slice.min_magnitude();
        assert_eq!(output, correct);

        // uniform stride but non-contiguous
        let slice = tensor.slice(s![.., .., 0]);

        let correct = NdArray::scalar(1).astype::<T>();
        let output = slice.min_magnitude();
        assert_eq!(output, correct);

        let correct = NdArray::new([1, 3, 4, 40]).astype::<T>();
        let output = slice.min_magnitude_along([1]);
        assert_eq!(output, correct);

        let correct = NdArray::new([1, 2]).astype::<T>();
        let output = slice.min_magnitude_along([0]);
        assert_eq!(output, correct);
    }
);

test_for_signed_dtypes!(
    test_reduce_max_magnitude, {
        let tensor = NdArray::new([
            [[-3, 5, 36], [2, 9, -4]],
            [[12, 56, 47], [3, 8, -36]],
            [[23, -67, 5], [-4, 7, 2]],
            [[-40, -80, 62], [-45, 6, -90]]
        ]).astype::<T>();

        // contiguous
        let correct = NdArray::scalar(90).astype::<T>();
        let output = tensor.max_magnitude();
        assert_eq!(output, correct);
        
        // non-uniform stride and non-contiguous
        let slice = tensor.slice(s![1..3, .., 0..=1]);
    
        let correct = NdArray::new([23, 67]).astype::<T>();
        let output = slice.max_magnitude_along([0, 1]);
        assert_eq!(output, correct);
    
        let correct = NdArray::scalar(67).astype::<T>();
        let output = slice.max_magnitude();
        assert_eq!(output, correct);
    
        // uniform stride but non-contiguous
        let slice = tensor.slice(s![.., .., 0]);
    
        let correct = NdArray::scalar(45).astype::<T>();
        let output = slice.max_magnitude();
        assert_eq!(output, correct);
    
        let correct = NdArray::new([3, 12, 23, 45]).astype::<T>();
        let output = slice.max_magnitude_along([1]);
        assert_eq!(output, correct);
    
        let correct = NdArray::new([40, 45]).astype::<T>();
        let output = slice.max_magnitude_along([0]);
        assert_eq!(output, correct);
    }
);

test_for_float_dtypes!(
    test_mean, {
        let tensor = NdArray::<T>::new([[1.0, 3.0], [2.0, 4.0], [3.0, 5.0]]);

        let correct = NdArray::<T>::new([2.0, 3.0, 4.0]);
        let output = tensor.mean_along(1);
        assert_eq!(output, correct);

        let correct = NdArray::<T>::new([[1.0, 3.0], [2.0, 4.0], [3.0, 5.0]]);
        let output = tensor.mean_along([]);
        assert_eq!(output, correct);

        let correct = NdArray::<T>::scalar(3.0);
        let output = tensor.mean();
        assert_eq!(output, correct);
    }
);

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
