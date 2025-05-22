use chela::*;
use paste::paste;


#[test]
#[should_panic]
fn test_diagonal_invalid_axis1() {
    let a = Tensor::arange(0, 12);
    let a = a.reshape([3, 4]);
    a.diagonal_along(3, 0);
}

#[test]
#[should_panic]
fn test_diagonal_invalid_axis2() {
    let a = Tensor::arange(0, 12);
    let a = a.reshape([3, 4]);
    a.diagonal_along(0, 3);
}

#[test]
#[should_panic]
fn test_diagonal_invalid_axes() {
    let a = Tensor::arange(0, 12);
    let a = a.reshape([3, 4]);
    a.diagonal_along(0, 0);
}

#[test]
#[should_panic]
fn test_dot_invalid_dimensions1() {
    let a = Tensor::arange(0, 12);
    let a = a.reshape([3, 4]);

    let b = Tensor::arange(0, 12);
    a.dot(b);
}

#[test]
#[should_panic]
fn test_dot_invalid_dimensions2() {
    let a = Tensor::arange(0, 12);
    let b = Tensor::arange(0, 12);
    let b = b.reshape([3, 4]);

    a.dot(b);
}

#[test]
#[should_panic]
fn test_dot_invalid_shapes() {
    let a = Tensor::arange(0, 12);
    let b = Tensor::arange(0, 11);
    a.dot(b);
}

test_for_all_numeric_dtypes!(
    test_dot_basic, {
        let a = Tensor::from([5, 8, 1, 2]).astype::<T>();
        let b = Tensor::from([4, 2, 7, 3]).astype::<T>();

        let expected = Tensor::scalar(20 + 16 + 7 + 6).astype::<T>();
        assert_eq!(a.dot(&b), expected);
        assert_eq!(b.dot(&a), expected);
        assert_eq!(a.dot(b), expected);
    }
);

test_for_common_numeric_dtypes!(
    test_dot, {
        for n in (1..30).step_by(3) {
            let a = (Tensor::arange(0, n) * 6).astype::<T>();
            let b = Tensor::arange(5, 5 + n).astype::<T>();

            let expected = Tensor::scalar((n - 1) * n * (2 * n + 14)).astype::<T>();
            assert_eq!(a.dot(&b), expected);
            assert_eq!(b.dot(&a), expected);
            assert_eq!(a.dot(b), expected);
        }
    }
);

test_for_common_numeric_dtypes!(
    test_matvec, {
        for m in 1..6 {
            for n in (1..30).step_by(3) {
                // Matrix: shape [m, n], values = 1 * 6, 2 * 6, ..., (m*n - 1) * 6
                let a = (Tensor::arange(0, m * n) * 6).reshape([m, n]).astype::<T>();

                // Vector: shape [n], values = [5, 6, ..., 5 + n - 1]
                let b = Tensor::arange(5, 5 + n).astype::<T>();

                // Compute matrix-vector product: [m] = [m, n] @ [n]
                let result = a.matmul(&b);

                // Manually compute expected result:
                let mut expected_data = vec![];
                for i in 0..m {
                    let mut sum = 0;
                    for j in 0..n {
                        let a_ij = (i * n + j) * 6;
                        let b_j = 5 + j;
                        sum += a_ij * b_j;
                    }
                    expected_data.push(sum);
                }

                let expected = Tensor::from(expected_data).astype::<T>();
                assert_eq!(result, expected);
            }
        }
    }
);


test_for_common_numeric_dtypes!(
    test_matvec_strided_views, {
        for m in 1..6 {
            for n in (1..30).step_by(3) {
                let a = (Tensor::arange(0, m * (n + 2) * 2) * 6)
                    .reshape([m, n + 2, 2])
                    .astype::<T>();
                let a = a.slice(s![.., 2.., 0]);
                assert_eq!(a.has_uniform_stride(), None);

                let b = Tensor::arange(0, n * 2)
                    .reshape([n, 2])
                    .astype::<T>();
                let b = b.slice_along(Axis(1), 0);
                assert!(!b.is_contiguous());

                let result = a.matmul(&b);
                let expected = einsum([&a, &b], (["ij", "j"], "i"));
                assert_eq!(result, expected);
            }
        }
    }
);



test_for_common_numeric_dtypes!(
    test_dot_mem_overlap, {
        for n in (1..30).step_by(3) {
            let a = Tensor::arange(0, n).astype::<T>();
            let b = a.view();

            let expected = Tensor::scalar((n - 1) * n * (2 * n - 1)).astype::<T>();
            assert_eq!(a.dot(&b) * Tensor::scalar(6).astype::<T>(), expected);
            assert_eq!(b.dot(&a), a.dot(&b));
            assert_eq!(a.dot(&b), a.dot(b));
        }
    }
);

test_for_all_numeric_dtypes!(
    test_diagonal, {
        let a = Tensor::arange(0, 12).astype::<T>();
        let a = a.reshape([3, 4]);

        let expected = Tensor::from([0, 5, 10]).astype::<T>();
        assert_eq!(a.diagonal(), expected);
        assert!(a.diagonal().is_view());

        assert_eq!(a.diagonal_along(0, 1), expected);
        assert_eq!(a.diagonal_along(-2, 1), expected);
        assert_eq!(a.offset_diagonal(0), expected);

        let expected = Tensor::from([1, 6, 11]).astype::<T>();
        assert_eq!(a.offset_diagonal(1), expected);

        let expected = Tensor::from([4, 9]).astype::<T>();
        assert_eq!(a.offset_diagonal(-1), expected);
    }
);

test_for_all_numeric_dtypes!(
    test_diagonal_3d, {
        let a = Tensor::arange(0, 8).astype::<T>();
        let a = a.reshape([2, 2, 2]);

        let expected = Tensor::from([[0, 6], [1, 7]]).astype::<T>();
        assert_eq!(a.diagonal(), expected);

        let expected = Tensor::from([[2], [3]]).astype::<T>();
        assert_eq!(a.offset_diagonal(1), expected);

        let expected = Tensor::from([[4], [5]]).astype::<T>();
        assert_eq!(a.offset_diagonal(-1), expected);

        let expected = Tensor::from([[0, 3], [4, 7]]).astype::<T>();
        assert_eq!(a.diagonal_along(1, 2), expected);
        assert_eq!(a.diagonal_along(2, 1), expected);

        let expected = Tensor::from([[1], [3]]).astype::<T>();
        assert_eq!(a.offset_diagonal_along(1, Axis(0), Axis(2)), expected);
    }
);

test_for_all_numeric_dtypes!(
    test_trace, {
        let a = Tensor::arange(0, 12).astype::<T>();
        let a = a.reshape([3, 4]);

        let expected = Tensor::scalar(15).astype::<T>();
        assert_eq!(a.trace(), expected);

        assert_eq!(a.trace_along(0, 1), expected);
        assert_eq!(a.trace_along(-2, 1), expected);
        assert_eq!(a.offset_trace(0), expected);

        let expected = Tensor::scalar(18).astype::<T>();
        assert_eq!(a.offset_trace(1), expected);

        let expected = Tensor::scalar(13).astype::<T>();
        assert_eq!(a.offset_trace(-1), expected);
    }
);


test_for_all_numeric_dtypes!(
    test_trace_3d, {
        let a = Tensor::arange(0, 8).astype::<T>();
        let a = a.reshape([2, 2, 2]);

        let expected = Tensor::from([6, 8]).astype::<T>();
        assert_eq!(a.trace(), expected);

        let expected = Tensor::from([2, 3]).astype::<T>();
        assert_eq!(a.offset_trace(1), expected);

        let expected = Tensor::from([4, 5]).astype::<T>();
        assert_eq!(a.offset_trace(-1), expected);

        let expected = Tensor::from([3, 11]).astype::<T>();
        assert_eq!(a.trace_along(1, 2), expected);
        assert_eq!(a.trace_along(2, 1), expected);

        let expected = Tensor::from([1, 3]).astype::<T>();
        assert_eq!(a.offset_trace_along(1, Axis(0), Axis(2)), expected);
    }
);
