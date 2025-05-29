use num::NumCast;
use chela::*;
use paste::paste;


#[test]
#[should_panic]
fn test_diagonal_invalid_axis1() {
    let a = NdArray::arange(0, 12);
    let a = a.reshape([3, 4]);
    a.diagonal_along(3, 0);
}

#[test]
#[should_panic]
fn test_diagonal_invalid_axis2() {
    let a = NdArray::arange(0, 12);
    let a = a.reshape([3, 4]);
    a.diagonal_along(0, 3);
}

#[test]
#[should_panic]
fn test_diagonal_invalid_axes() {
    let a = NdArray::arange(0, 12);
    let a = a.reshape([3, 4]);
    a.diagonal_along(0, 0);
}

#[test]
#[should_panic]
fn test_dot_invalid_dimensions1() {
    let a = NdArray::arange(0, 12);
    let a = a.reshape([3, 4]);

    let b = NdArray::arange(0, 12);
    a.dot(b);
}

#[test]
#[should_panic]
fn test_dot_invalid_dimensions2() {
    let a = NdArray::arange(0, 12);
    let b = NdArray::arange(0, 12);
    let b = b.reshape([3, 4]);

    a.dot(b);
}

#[test]
#[should_panic]
fn test_dot_invalid_shapes() {
    let a = NdArray::arange(0, 12);
    let b = NdArray::arange(0, 11);
    a.dot(b);
}

#[test]
#[should_panic]
fn test_matvec_matrix_not_2d() {
    let a = NdArray::arange(0, 12);
    let a = a.reshape([3, 2, 2]);

    let b = NdArray::arange(0, 2);

    a.matmul(&b);
}

#[test]
#[should_panic]
fn test_matvec_vector_not_1d() {
    let a = NdArray::arange(0, 6);
    let a = a.reshape([2, 3]);

    let b = NdArray::arange(0, 6);
    let b = b.reshape([2, 3]);

    a.matmul(&b);
}

#[test]
#[should_panic]
fn test_matvec_inner_dim_mismatch() {
    let a = NdArray::arange(0, 6);
    let a = a.reshape([2, 3]);

    let b = NdArray::arange(0, 4);

    a.matmul(&b);
}

#[test]
#[should_panic]
fn test_matmat_matrix1_not_2d() {
    let a = NdArray::arange(0, 12);
    let a = a.reshape([3, 2, 2]);

    let b = NdArray::arange(0, 6);
    let b = b.reshape([2, 3]);

    a.matmul(&b);
}

#[test]
#[should_panic]
fn test_matmat_matrix2_not_2d() {
    let a = NdArray::arange(0, 6);
    let a = a.reshape([2, 3]);

    let b = NdArray::arange(0, 12);
    let b = b.reshape([2, 2, 3]);

    a.matmul(&b);
}

#[test]
#[should_panic]
fn test_matmat_inner_dim_mismatch() {
    let a = NdArray::arange(0, 6);
    let a = a.reshape([2, 3]);

    let b = NdArray::arange(0, 8);
    let b = b.reshape([4, 2]);
    a.matmul(&b);
}

test_for_all_numeric_dtypes!(
    test_dot_basic, {
        let a = NdArray::from([5, 8, 1, 2]).astype::<T>();
        let b = NdArray::from([4, 2, 7, 3]).astype::<T>();

        let expected = NdArray::scalar(20 + 16 + 7 + 6).astype::<T>();
        assert_eq!(a.dot(&b), expected);
        assert_eq!(b.dot(&a), expected);
        assert_eq!(a.dot(b), expected);
    }
);

test_for_common_numeric_dtypes!(
    test_dot, {
        for n in (1..30).step_by(3) {
            let a = (NdArray::arange(0, n) * 6).astype::<T>();
            let b = NdArray::arange(5, 5 + n).astype::<T>();

            let expected = NdArray::scalar((n - 1) * n * (2 * n + 14)).astype::<T>();
            assert_eq!(a.dot(&b), expected);
            assert_eq!(b.dot(&a), expected);
            assert_eq!(a.matmul(&b), expected);
            assert_eq!(a.dot(b), expected);
        }
    }
);

test_for_common_numeric_dtypes!(
    test_matvec, {
        for m in (1..28).step_by(9) {
            for n in (1..30).step_by(3) {
                let a = (NdArray::arange(0, m * n) * 6)
                    .reshape([m, n])
                    .astype::<T>();

                let b = NdArray::arange(5, 5 + n)
                    .astype::<T>();

                let result = a.matmul(&b);
                let expected = einsum([&a, &b], (["ij", "j"], "i"));
                assert_eq!(result, expected);
            }
        }
    }
);


test_for_common_numeric_dtypes!(
    test_matvec_strided_views, {
        for m in (1..28).step_by(9) {
            for n in (1..30).step_by(3) {
                let a = (NdArray::arange(0, m * (n + 2) * 2) * 6)
                    .reshape([m, n + 2, 2])
                    .astype::<T>();
                let a = a.slice(s![.., 2.., 0]);
                assert_eq!(a.has_uniform_stride(), None);

                let b = NdArray::arange(0, n * 2)
                    .reshape([n, 2])
                    .astype::<T>();
                let b = b.slice_along(Axis(1), 0);
                assert!(!b.is_contiguous());

                let expected = einsum([&a, &b], (["ij", "j"], "i"));
                assert_eq!(a.matmul(&b), expected);
                assert_eq!(a.matmul(b), expected);
            }
        }
    }
);

test_for_common_numeric_dtypes!(
    test_matmat, {
        for m in (1..25).step_by(6) {
            for k in (1..28).step_by(9) {
                for n in (1..28).step_by(9) {
                    let a = (NdArray::arange(0, m * k) * 3)
                        .reshape([m, k])
                        .astype::<T>();

                    let b = (NdArray::arange(0, k * n) * 7)
                        .reshape([k, n])
                        .astype::<T>();

                    let expected = einsum([&a, &b], (["ik", "kj"], "ij"));
                    assert_eq!(a.matmul(&b), expected);
                    assert_eq!(a.matmul(b), expected);
                }
            }
        }
    }
);

test_for_common_numeric_dtypes!(
    test_batched_matmat, {
        let high = <T as NumCast>::from(10).unwrap();

        for b in (1..37).step_by(9) {
            for m in (1..28).step_by(9) {
                for k in (1..21).step_by(5) {
                    for n in (1..25).step_by(6) {
                        let lhs = NdArray::<T>::randint([b, m, k], T::default(), high);
                        let rhs = NdArray::<T>::randint([b, k, n], T::default(), high);

                        let expected = einsum([&lhs, &rhs], (["bik", "bkj"], "bij"));
                        assert_almost_eq!(lhs.bmm(&rhs), expected, 0.1);
                    }
                }
            }
        }
    }
);

test_for_common_numeric_dtypes!(
    test_matmat_strided_views, {
        for m in (1..37).step_by(9) {
            for k in (1..21).step_by(4) {
                for n in (1..25).step_by(6) {
                    let a = (NdArray::arange(0, m * (k + 3) * 2) * 6)
                        .reshape([m, k + 3, 2])
                        .astype::<T>();
                    let a = a.slice(s![.., 2..k + 2, 0]);
                    assert_eq!(a.shape(), &[m, k]);
                    assert_eq!(a.has_uniform_stride(), None);

                    let b = (NdArray::arange(0, k * (n + 2) * 2) * 7)
                        .reshape([k, n + 2, 2])
                        .astype::<T>();
                    let b = b.slice(s![.., ..n, 0]);
                    assert_eq!(b.shape(), &[k, n]);
                    assert!(!b.is_contiguous());

                    let result = a.matmul(&b);
                    let expected = einsum([&a, &b], (["ik", "kj"], "ij"));
                    assert_eq!(result, expected);
                }
            }
        }
    }
);

test_for_common_numeric_dtypes!(
    test_dot_mem_overlap, {
        for n in (1..31).step_by(5) {
            let a = NdArray::arange(0, n).astype::<T>();
            let b = (&a).view();

            let expected = NdArray::scalar((n - 1) * n * (2 * n - 1)).astype::<T>();
            assert_eq!(a.dot(&b) * NdArray::scalar(6).astype::<T>(), expected);
            assert_eq!(b.dot(&a), a.dot(&b));
            assert_eq!(a.dot(&b), a.dot(b));
        }
    }
);

test_for_all_numeric_dtypes!(
    test_diagonal, {
        let a = NdArray::arange(0, 12).astype::<T>();
        let a = a.reshape([3, 4]);

        let expected = NdArray::from([0, 5, 10]).astype::<T>();
        assert_eq!(a.diagonal(), expected);
        assert!(a.diagonal().is_view());

        assert_eq!(a.diagonal_along(0, 1), expected);
        assert_eq!(a.diagonal_along(-2, 1), expected);
        assert_eq!(a.offset_diagonal(0), expected);

        let expected = NdArray::from([1, 6, 11]).astype::<T>();
        assert_eq!(a.offset_diagonal(1), expected);

        let expected = NdArray::from([4, 9]).astype::<T>();
        assert_eq!(a.offset_diagonal(-1), expected);
    }
);

test_for_all_numeric_dtypes!(
    test_diagonal_3d, {
        let a = NdArray::arange(0, 8).astype::<T>();
        let a = a.reshape([2, 2, 2]);

        let expected = NdArray::from([[0, 6], [1, 7]]).astype::<T>();
        assert_eq!(a.diagonal(), expected);

        let expected = NdArray::from([[2], [3]]).astype::<T>();
        assert_eq!(a.offset_diagonal(1), expected);

        let expected = NdArray::from([[4], [5]]).astype::<T>();
        assert_eq!(a.offset_diagonal(-1), expected);

        let expected = NdArray::from([[0, 3], [4, 7]]).astype::<T>();
        assert_eq!(a.diagonal_along(1, 2), expected);
        assert_eq!(a.diagonal_along(2, 1), expected);

        let expected = NdArray::from([[1], [3]]).astype::<T>();
        assert_eq!(a.offset_diagonal_along(1, Axis(0), Axis(2)), expected);
    }
);

test_for_all_numeric_dtypes!(
    test_trace, {
        let a = NdArray::arange(0, 12).astype::<T>();
        let a = a.reshape([3, 4]);

        let expected = NdArray::scalar(15).astype::<T>();
        assert_eq!(a.trace(), expected);

        assert_eq!(a.trace_along(0, 1), expected);
        assert_eq!(a.trace_along(-2, 1), expected);
        assert_eq!(a.offset_trace(0), expected);

        let expected = NdArray::scalar(18).astype::<T>();
        assert_eq!(a.offset_trace(1), expected);

        let expected = NdArray::scalar(13).astype::<T>();
        assert_eq!(a.offset_trace(-1), expected);
    }
);


test_for_all_numeric_dtypes!(
    test_trace_3d, {
        let a = NdArray::arange(0, 8).astype::<T>();
        let a = a.reshape([2, 2, 2]);

        let expected = NdArray::from([6, 8]).astype::<T>();
        assert_eq!(a.trace(), expected);

        let expected = NdArray::from([2, 3]).astype::<T>();
        assert_eq!(a.offset_trace(1), expected);

        let expected = NdArray::from([4, 5]).astype::<T>();
        assert_eq!(a.offset_trace(-1), expected);

        let expected = NdArray::from([3, 11]).astype::<T>();
        assert_eq!(a.trace_along(1, 2), expected);
        assert_eq!(a.trace_along(2, 1), expected);

        let expected = NdArray::from([1, 3]).astype::<T>();
        assert_eq!(a.offset_trace_along(1, Axis(0), Axis(2)), expected);
    }
);
