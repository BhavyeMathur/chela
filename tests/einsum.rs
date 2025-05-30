use chela::*;
use paste::paste;
use rand_distr::num_traits::NumCast;

#[test]
#[should_panic]
fn test_einsum_non_ascii_input() {
    let a = NdArray::from([[1, 2], [0, 0]]);
    let b = NdArray::from([[3, 4], [5, 5]]);
    let _ = einsum([&a, &b], (["±i", "i-"], "i"));
}

#[test]
#[should_panic]
fn test_einsum_non_ascii_output() {
    let a = NdArray::from([[1, 2], [0, 0]]);
    let b = NdArray::from([[3, 4], [5, 5]]);
    let _ = einsum([&a, &b], (["ij", "jk"], "i±"));
}

#[test]
#[should_panic]
fn test_einsum_non_letters() {
    let a = NdArray::from([[1, 2], [0, 0]]);
    let b = NdArray::from([[3, 4], [5, 5]]);
    let _ = einsum([&a, &b], (["01", "12"], "02"));
}

#[test]
#[should_panic]
fn test_einsum_invalid_input_labels() {
    let a = NdArray::from([[1, 2], [0, 0]]);
    let b = NdArray::from([[3, 4], [5, 5]]);
    let _ = einsum([&a, &b], (["i", "jk"], "ik"));
}

#[test]
#[should_panic]
fn test_einsum_invalid_output_labels() {
    let a = NdArray::from([[1, 2], [0, 0]]);
    let b = NdArray::from([[3, 4], [5, 5]]);
    let _ = einsum([&a, &b], (["i", "jk"], "02"));
}

#[test]
#[should_panic]
fn test_einsum_dimension_mismatch() {
    let a = NdArray::from([[1, 2]]);
    let b = NdArray::from([[3, 4], [5, 6], [7, 8]]);
    let _ = einsum([&a, &b], (["ij", "jk"], "ik"));
}

#[test]
#[should_panic]
fn test_einsum_invalid_index() {
    let a = NdArray::from([[1, 2]]);
    let b = NdArray::from([[3, 4], [5, 6], [7, 8]]);
    let _ = einsum([&a, &b], (["ij", "kl"], "m"));
}


test_for_common_numeric_dtypes!(
    test_einsum_sums, {
        for n in 1..17 {
            // sum
            let a = NdArray::arange(0, n).astype::<T>();
            let expected = a.sum();
            let result = chela::einsum([&a], (["i"], ""));
            assert_almost_eq!(result, expected);

            // trace
            let a = NdArray::arange(0, n * n).astype::<T>();
            let a = a.reshape([n, n]);
            let result = chela::einsum([&a], (["ii"], ""));
            let expected = a.trace();
            assert_almost_eq!(result, expected);
        }
    }
);

test_for_common_numeric_dtypes!(
    test_einsum_sum_slice, {
        for n in (100..2000).step_by(100) {
            let a = NdArray::arange(0, 2 * n).astype::<T>();
            let a = a.reshape([n, 2]);
            let a = a.slice_along(Axis(1), 0);
            let a = a.reshape([n]);

            let expected = a.sum();
            let result = chela::einsum([&a], (["i"], ""));
            assert_almost_eq!(result, expected);
        }

        for n in (1..50).step_by(5) {
            let a = NdArray::arange(0, 2 * n * n).astype::<T>();
            let a = a.reshape([n, 2, n]);
            let a = a.slice_along(Axis(1), 0);

            let expected = a.sum();
            let result = chela::einsum([&a], (["ij"], ""));
            assert_almost_eq!(result, expected);

            let expected = a.sum_along(Axis(0));
            let result = chela::einsum([&a], (["ij"], "j"));
            assert_almost_eq!(result, expected);

            let expected = a.sum_along(Axis(1));
            let result = chela::einsum([&a], (["ij"], "i"));
            assert_almost_eq!(result, expected);
        }

        for n in (1..50).step_by(5) {
            let a = NdArray::arange(0, 2 * 2 * n * n).astype::<T>();
            let a = a.reshape([n, 2, n, 2]);
            let a = a.slice(s!(.., 0, .., 0));

            let expected = a.sum();
            let result = chela::einsum([&a], (["ij"], ""));
            assert_almost_eq!(result, expected);

            let expected = a.sum_along(Axis(0));
            let result = chela::einsum([&a], (["ij"], "j"));
            assert_almost_eq!(result, expected);

            let expected = a.sum_along(Axis(1));
            let result = chela::einsum([&a], (["ij"], "i"));
            assert_almost_eq!(result, expected);
        }

        for n in (3..24).step_by(5) {
            let a = NdArray::arange(0, 2 * 2 * n * n).astype::<T>();
            let a = a.reshape([n, 2, n, 2]);
            let a = a.slice(s!(.., .., .., 0));

            let expected = a.sum();
            let result = chela::einsum([&a], (["ijk"], ""));
            assert_almost_eq!(result, expected);

            let expected = a.sum_along(Axis(0));
            let result = chela::einsum([&a], (["ijk"], "jk"));
            assert_almost_eq!(result, expected);

            let expected = a.sum_along(Axis(1));
            let result = chela::einsum([&a], (["ijk"], "ik"));
            assert_almost_eq!(result, expected);

            let expected = a.sum_along(Axis(2));
            let result = chela::einsum([&a], (["ijk"], "ij"));
            assert_almost_eq!(result, expected);

            let expected = a.sum_along(Axis(2));
            let expected = expected.sum_along(Axis(1));
            let result = chela::einsum([&a], (["ijk"], "i"));
            assert_almost_eq!(result, expected);

            let expected = a.sum_along(Axis(2));
            let expected = expected.sum_along(Axis(0));
            let result = chela::einsum([&a], (["ijk"], "j"));
            assert_almost_eq!(result, expected);

            let expected = a.sum_along(Axis(1));
            let expected = expected.sum_along(Axis(0));
            let result = chela::einsum([&a], (["ijk"], "k"));
            assert_almost_eq!(result, expected);
        }
    }
);

test_for_common_numeric_dtypes!(
    test_einsum_matmul, {
        let a = NdArray::from([[1, 2], [3, 4]]).astype::<T>();
        let b = NdArray::from([[5, 6], [7, 8]]).astype::<T>();

        let expected = NdArray::from([[19, 22], [43, 50]]).astype::<T>();
        let result = chela::einsum([&a, &b], (["ij", "jk"], "ik"));
        assert_almost_eq!(result, expected);

        // bigger matmul
        let n: usize = 20;
        let a = NdArray::arange(0, n * n).astype::<T>();
        let b = NdArray::arange(n, n + n * n).astype::<T>();
        let a = a.reshape([n, n]);
        let b = b.reshape([n, n]);

        let mut expected_data = vec![T::default(); n * n];
        for i in 0..n {
            for k in 0..n {
                let mut sum = T::default();
                for j in 0..n {
                    sum = sum + a[[i, j]] * b[[j, k]];
                }
                expected_data[i * n + k] = sum;
            }
        }

        let expected = NdArray::from(expected_data);
        let expected = expected.reshape([n, n]);

        let result = chela::einsum(&[&a, &b], (&["ij", "jk"], "ik"));
        assert_almost_eq!(result, expected);
    }
);

test_for_common_numeric_dtypes!(
    test_einsum_matrix_vector, {
        let n: usize = 20;
        let a = NdArray::arange(0, n * n).astype::<T>();
        let b = NdArray::arange(n, 2 * n).astype::<T>();
        let a = a.reshape([n, n]);

        let expected = {
            let mut out = vec![T::default(); n];

            for i in 0..n {
                for j in 0..n {
                    out[i] += a[[i, j]] * b[j];
                }
            }

            NdArray::from(out)
        };

        let result = chela::einsum(&[&a, &b], (&["ij", "j"], "i"));
        assert_almost_eq!(result, expected);
    }
);

test_for_all_numeric_dtypes!(
    test_einsum_pointwise_multiplication, {
        let a = NdArray::from([[1, 2, 3], [0, 1, 2], [4, 5, 6]]).astype::<T>();
        let b = NdArray::from([[5, 6, 7], [10, 20, 30], [3, 6, 9]]).astype::<T>();

        let expected = NdArray::from([[5, 12, 21], [0, 20, 60], [12, 30, 54]]).astype::<T>();
        let result = chela::einsum([&a, &b], (&["ij", "ij"], "ij"));
        assert_almost_eq!(result, expected);

        // larger pointwise multiplication
        let a = NdArray::arange(0i32, 20).astype::<T>();
        let b = NdArray::arange_with_step(19i32, -1, -1).astype::<T>();
        let a = a.reshape([2, 10]);
        let b = b.reshape([2, 10]);

        let expected_data: Vec<T> = a.flatiter().zip(b.flatiter()).map(|(x, y)| x * y).collect();
        let expected = NdArray::from(expected_data);
        let expected = expected.reshape([2, 10]);

        let result = chela::einsum(&[&a, &b], (["ij", "ij"], "ij"));
        assert_almost_eq!(result, expected);
    }
);

test_for_common_numeric_dtypes!(
    test_einsum_ij_ki_j, {
        let a = NdArray::arange(0i32, 20).astype::<T>();
        let b = NdArray::arange_with_step(19i32, -1, -1).astype::<T>();
        let a = a.reshape([2, 10]);
        let b = b.reshape([10, 2]);

        let expected = {
            let mut out = vec![T::default(); 10];

            for i in 0..2 {
                for j in 0..10 {
                    for k in 0..10 {
                        out[j] += a[[i, j]] * b[[k, i]];
                    }
                }
            }

            NdArray::from(out)
        };

        let result = chela::einsum(&[&a, &b], (["ij", "ki"], "j"));
        assert_almost_eq!(result, expected);
    }
);

test_for_common_numeric_dtypes!(
    test_einsum_ij_ki_i, {
        let a = NdArray::arange(0i32, 46).astype::<T>();
        let b = NdArray::arange_with_step(19i32, -1, -1).astype::<T>();
        let a = a.reshape([2, 23]);
        let b = b.reshape([10, 2]);

        let expected = {
            let mut out = vec![T::default(); 2];

            for i in 0..2 {
                for j in 0..23 {
                    for k in 0..10 {
                        out[i] += a[[i, j]] * b[[k, i]];
                    }
                }
            }

            NdArray::from(out)
        };

        let result = chela::einsum(&[&a, &b], (["ij", "ki"], "i"));
        assert_almost_eq!(result, expected);
    }
);

test_for_all_numeric_dtypes!(
    test_einsum_sum_along_axes, {
        let a = NdArray::from([[1, 2], [0, 1]]).astype::<T>();
        let b = NdArray::from([[5, 6], [10, 20]]).astype::<T>();

        let expected = NdArray::from([71, 30]).astype::<T>();
        let result = chela::einsum([&a, &b], (["ij", "jk"], "i"));
        assert_almost_eq!(result, expected);

        let expected = NdArray::from([11, 90]).astype::<T>();
        let result = chela::einsum([&a, &b], (["ij", "jk"], "j"));
        assert_almost_eq!(result, expected);

        let a = NdArray::from([[1, 2, 3], [4, 5, 6]]).astype::<T>();
        let result = einsum([&a], (["ij"], "i"));
        let expected = NdArray::from([6, 15]).astype::<T>();
        assert_almost_eq!(result, expected);
    }
);

test_for_common_numeric_dtypes!(
    test_einsum_sum_along_axes_big, {
        let n: usize = 18;
        let a = NdArray::arange(0, n * n).astype::<T>();
        let b = NdArray::arange(0, n * n).astype::<T>();
        let a = a.reshape([n, n]);
        let b = b.reshape([n, n]);

        let expected = {
            let mut out = vec![T::default(); n];

            for i in 0..n {
                for j in 0..n {
                    for k in 0..n {
                        out[i] += a[[i, j]] * b[[j, k]];
                    }
                }
            }

            NdArray::from(out)
        };

        let result = chela::einsum([&a, &b], (["ij", "jk"], "i"));
        assert_almost_eq!(result, expected);
    }
);

test_for_all_numeric_dtypes!(
    test_einsum_sum_product, {
        let a = NdArray::from([[1, 2], [0, 1]]).astype::<T>();
        let b = NdArray::from([[5, 6], [10, 20]]).astype::<T>();

        let expected = NdArray::scalar(63).astype::<T>();
        let result = chela::einsum([&a, &b], (["ij", "ik"], ""));
        assert_almost_eq!(result, expected);

        let expected = NdArray::scalar(101).astype::<T>();
        let result = chela::einsum([&a, &b], (["ij", "jk"], ""));
        assert_almost_eq!(result, expected);

        let expected = NdArray::scalar(71).astype::<T>();
        let result = chela::einsum([&a, &b], (["ij", "ki"], ""));
        assert_almost_eq!(result, expected);

        let expected = NdArray::scalar(93).astype::<T>();
        let result = chela::einsum([&a, &b], (["ij", "kj"], ""));
        assert_almost_eq!(result, expected);
    }
);

test_for_common_integer_dtypes!(
    test_einsum_sum_product_slice, {
        let n = 17;

        let a = NdArray::arange(0, 2 * 2 * n * n).astype::<T>();
        let a = a.reshape([n, 2, n, 2]);
        let a = a.slice(s!(.., 0, .., 0));

        let b = NdArray::arange(0, 2 * 2 * n * n).astype::<T>();
        let b = b.reshape([n, 2, n, 2]);
        let b = b.slice(s!(.., 0, .., 0));

        let expected = {
            let mut out = T::default();

            for i in 0..n {
                for j in 0..n {
                    for k in 0..n {
                        out += a[[i, j]] * b[[j, k]];
                    }
                }
            }

            NdArray::scalar(out)
        };
        let result = chela::einsum([&a, &b], (["ij", "jk"], ""));
        assert_almost_eq!(result, expected);
    }
);

test_for_float_dtypes!(
    test_einsum_sum_product_slice_float, {
        let n = 23;

        let a = NdArray::arange(0, 2 * 2 * n * n).astype::<T>();
        let a = a.reshape([n, 2, n, 2]);
        let a = a.slice(s!(.., 0, .., 0)) * 0.001;

        let b = NdArray::arange(0, 2 * 2 * n * n).astype::<T>();
        let b = b.reshape([n, 2, n, 2]);
        let b = b.slice(s!(.., 0, .., 0)) * 0.001;

        let expected = {
            let mut out = T::default();

            for i in 0..n {
                for j in 0..n {
                    for k in 0..n {
                        out += a[[i, j]] * b[[j, k]];
                    }
                }
            }

            NdArray::scalar(out)
        };
        let result = chela::einsum([&a, &b], (["ij", "jk"], ""));
        assert_almost_eq!(result, expected, 0.01);
    }
);

test_for_float_dtypes!(
    test_einsum_sum_product_on_slice_float, {
        let n = 23;

        let a = NdArray::arange(0, n * n).astype::<T>();
        let b = NdArray::arange(0, 2 * n * n).astype::<T>();
        let a = a.reshape([n, n]) / ( (n * n) as T);
        let b = b.reshape([n, n, 2]) / ( (n * n) as T);

        let b = b.slice_along(Axis(2), 0);

        let expected = {
            let mut out = T::default();

            for i in 0..n {
                for j in 0..n {
                    for k in 0..n {
                        out += a[[i, j]] * b[[j, k]];
                    }
                }
            }

            NdArray::scalar(out)
        };

        let result = einsum([&a, &b], (["ij", "jk"], ""));
        assert_almost_eq!(result, expected, 0.01);
    }
);

test_for_common_integer_dtypes!(
    test_einsum_sum_product_on_slice, {
        let n = 23;

        let a = NdArray::arange(0, n * n).astype::<T>();
        let b = NdArray::arange(0, 2 * n * n).astype::<T>();
        let a = a.reshape([n, n]);
        let b = b.reshape([n, n, 2]);

        let b = b.slice_along(Axis(2), 0);

        let expected = {
            let mut out = T::default();

            for i in 0..n {
                for j in 0..n {
                    for k in 0..n {
                        out += a[[i, j]] * b[[j, k]];
                    }
                }
            }

            NdArray::scalar(out)
        };

        let result = chela::einsum([&a, &b], (["ij", "jk"], ""));
        assert_almost_eq!(result, expected);
    }
);

test_for_common_integer_dtypes!(
    test_einsum_sum_product_big, {
        let n: usize = 23;
        let a = NdArray::arange(0, n * n).astype::<T>();
        let b = NdArray::arange(0, n * n).astype::<T>();
        let a = a.reshape([n, n]);
        let b = b.reshape([n, n]);

        let expected = {
            let mut out = T::default();

            for i in 0..n {
                for j in 0..n {
                    for k in 0..n {
                        out += a[[i, j]] * b[[k, j]];
                    }
                }
            }

            NdArray::scalar(out)
        };

        let result = chela::einsum([&a, &b], (["ij", "kj"], ""));
        assert_almost_eq!(result, expected);
    }
);

test_for_float_dtypes!(
    test_einsum_sum_product_big_float, {
        let n: usize = 23;
        let a = NdArray::arange(0, n * n).astype::<T>() * 0.01;
        let b = NdArray::arange(0, n * n).astype::<T>() * 0.01;
        let a = a.reshape([n, n]);
        let b = b.reshape([n, n]);

        let expected = {
            let mut out = T::default();

            for i in 0..n {
                for j in 0..n {
                    for k in 0..n {
                        out += a[[i, j]] * b[[k, j]];
                    }
                }
            }

            NdArray::scalar(out)
        };

        let result = chela::einsum([&a, &b], (["ij", "kj"], ""));
        assert_almost_eq!(result, expected, 0.2);
    }
);

test_for_all_numeric_dtypes!(
    test_einsum_2operands_to_3d, {
        let a = NdArray::from([[1, 2], [0, 1]]).astype::<T>();
        let b = NdArray::from([[5, 6], [10, 20]]).astype::<T>();

        let expected = NdArray::from([[[5, 10], [12, 40]], [[0, 0], [6, 20]]]).astype::<T>();
        let result = chela::einsum([&a, &b], (["ij", "kj"], "ijk"));
        assert_almost_eq!(result, expected);

        let expected = NdArray::from([[[5, 6], [10, 12]], [[0, 0], [10, 20]]]).astype::<T>();
        let result = chela::einsum([&a, &b], (["ij", "ik"], "ijk"));
        assert_almost_eq!(result, expected);
    }
);

test_for_all_numeric_dtypes!(
    test_einsum_inner_product, {
        let a = NdArray::from([1, 2, 3]).astype::<T>();
        let b = NdArray::from([4, 5, 6]).astype::<T>();

        let expected = NdArray::scalar(32).astype::<T>();
        let result = einsum([&a, &b], (["i", "i"], ""));
        assert_almost_eq!(result, expected);
    }
);

test_for_all_numeric_dtypes!(
    test_einsum_outer_product, {
        let a = NdArray::from([1, 2]).astype::<T>();
        let b = NdArray::from([3, 4, 5]).astype::<T>();

        let expected = NdArray::from([[3, 4, 5], [6, 8, 10]]).astype::<T>();
        let result = einsum([&a, &b], (["i", "j"], "ij"));
        assert_almost_eq!(result, expected);
    }
);

test_for_all_numeric_dtypes!(
    test_einsum_matrix_outer_product, {
        let a = NdArray::from([[1, 2], [3, 4]]).astype::<T>();
        let b = NdArray::from([[5, 6], [7, 8]]).astype::<T>();

        let expected = NdArray::from([
            [[[5, 6], [7, 8]], [[10, 12], [14, 16]]],
            [[[15, 18], [21, 24]], [[20, 24], [28, 32]]]
        ]).astype::<T>();

        let result = chela::einsum([&a, &b], (["ij", "kl"], "ijkl"));
        assert_almost_eq!(result, expected);
    }
);

test_for_all_numeric_dtypes!(
    test_einsum_three_operands,
    {
        let a = NdArray::from([[1, 2], [3, 4], [5, 1]]).astype::<T>();
        let b = NdArray::from([[5, 6, 2, 2], [7, 8, 1, 0]]).astype::<T>();
        let c = NdArray::from([[1, 0], [0, 1], [0, 2], [2, 0]]).astype::<T>();

        let expected = einsum([&einsum([&a, &b], (["ij", "jk"], "ik")), &c], (["ik", "kl"], "il"));
        let result = einsum([&a, &b, &c], (["ij", "jk", "kl"], "il"));
        assert_almost_eq!(result, expected);
    }
);

test_for_common_numeric_dtypes!(
    test_einsum_three_operands_big, {
        let n = 20;
        let a = NdArray::arange(0, n).astype::<T>();
        let b = NdArray::arange(0, n).astype::<T>();
        let c = NdArray::arange(0, n).astype::<T>();

        let expected = {
            let mut out = T::default();

            for i in 0..n {
                for j in 0..n {
                    for k in 0..n {
                        out += a[i] * b[j] * c[k];
                    }
                }
            }

            NdArray::scalar(out)
        };
        let result = einsum([&a, &b, &c], (["i", "j", "k"], ""));
        assert_almost_eq!(result, expected);
    }
);

test_for_all_numeric_dtypes!(
    test_einsum_scalar_times_tensor,
    {
        let a = NdArray::from([[1, 2], [3, 4]]).astype::<T>();
        let b = NdArray::scalar(10).astype::<T>();

        let expected = NdArray::from([[10, 20], [30, 40]]).astype::<T>();
        let result = einsum([&a, &b], (["ij", ""], "ij"));
        assert_almost_eq!(result, expected);
    }
);

test_for_all_numeric_dtypes!(
    test_einsum_transpose,
    {
        let a = NdArray::from([[1, 2, 3], [4, 5, 6]]).astype::<T>();
        let expected = NdArray::from([[1, 4], [2, 5], [3, 6]]).astype::<T>();

        let result = einsum([&a], (["ij"], "ji"));
        assert_almost_eq!(result, expected);
        assert_almost_eq!(result, (&a).T());

        let result = einsum_view(&a, ("ij", "ji")).unwrap();
        assert_almost_eq!(result, expected);
        assert_almost_eq!(result, (&a).T());
    }
);

test_for_all_numeric_dtypes!(
    test_einsum_broadcasting_vector_matrix,
    {
        let a = NdArray::from([1, 2]).astype::<T>();
        let b = NdArray::from([[3, 4, 5], [6, 7, 8]]).astype::<T>();
        let result = einsum([&a, &b], (["i", "ij"], "ij"));
        let expected = NdArray::from([[3, 4, 5], [12, 14, 16]]).astype::<T>();
        assert_almost_eq!(result, expected);

        let b = NdArray::from([[3, 4], [5, 6]]).astype::<T>();
        let result = einsum([&a, &b], (["i", "ij"], "ij"));
        let expected = NdArray::from([[3, 4], [10, 12]]).astype::<T>();
        assert_almost_eq!(result, expected);
    }
);

test_for_all_numeric_dtypes!(
    test_einsum_diagonal_extraction,
    {
        let a = NdArray::from([[1, 2, 3], [4, 5, 6], [7, 8, 9]]).astype::<T>();
        let expected = a.diagonal();

        let result = einsum([&a], (["ii"], "i"));
        assert_almost_eq!(result, expected);

        let result = einsum_view(&a, ("ii", "i")).unwrap();
        assert_almost_eq!(result, expected);
    }
);

test_for_all_numeric_dtypes!(
    test_einsum_tensor_contraction,
    {
        let a = NdArray::from([[[1, 2], [3, 4]]]).astype::<T>();
        let b = NdArray::from([[5, 6], [7, 8]]).astype::<T>();
        let result = einsum([&a, &b], (["ijk", "kl"], "ijl"));
        let expected = NdArray::from([[[19, 22], [43, 50]]]).astype::<T>();
        assert_almost_eq!(result, expected);
    }
);

test_for_all_numeric_dtypes!(
    test_einsum_all_sum,
    {
        let a = NdArray::from([[1, 2], [3, 4]]).astype::<T>();
        
        let expected = NdArray::scalar(10).astype::<T>();
        let result = einsum([&a], (["ij"], ""));
        assert_almost_eq!(result, expected);
    }
);

test_for_all_numeric_dtypes!(
    test_einsum_identity,
    {
        let a = NdArray::from([[9, 8], [7, 6]]).astype::<T>();
        let result = einsum([&a], (["ij"], "ij"));
        assert_almost_eq!(result, a);

        let a = NdArray::from([[0, 0], [0, 0]]).astype::<T>();
        let result = einsum([&a], (["ij"], "ij"));
        assert_almost_eq!(result, a);

        let result = einsum_view(&a, ("ij", "ij")).unwrap();
        assert_almost_eq!(result, a);
    }
);

test_for_all_numeric_dtypes!(
    test_einsum_batch_matmul,
    {
        let a = NdArray::from([[[1, 2], [3, 4]]]).astype::<T>();
        let b = NdArray::from([[[5, 6], [7, 8]]]).astype::<T>();
        let result = einsum([&a, &b], (["bij", "bjk"], "bik"));
        let expected = NdArray::from([[[19, 22], [43, 50]]]).astype::<T>();
        assert_almost_eq!(result, expected);
    }
);

test_for_common_numeric_dtypes!(
    test_einsum_batched_matmul, {
        let low = T::default();
        let high = <T as NumCast>::from(10).unwrap();

        for b in (1..37).step_by(9) {
            for m in 1..4 {
                for k in (1..25).step_by(8) {
                    for n in (1..25).step_by(6) {

                        let lhs = NdArray::<T>::randint([b, m, k], low, high);
                        let rhs = NdArray::<T>::randint([b, k, n], low, high);
                        let result = einsum([&lhs, &rhs], (["bik", "bkj"], "bij"));

                        let mut expected_data = vec![];
                        for b in 0..b {
                            for i in 0..m {
                                for j in 0..n {
                                    let mut sum = T::default();
                                    for kk in 0..k {
                                        let a_val = lhs[[b, i, kk]];
                                        let b_val = rhs[[b, kk, j]];
                                        sum += a_val * b_val;
                                    }
                                    expected_data.push(sum);
                                }
                            }
                        }

                        let expected = NdArray::from(expected_data)
                            .reshape([b, m, n])
                            .astype::<T>();

                        assert_eq!(result, expected);
                    }
                }
            }
        }
    }
);

// #[test]
// fn test_einsum_repeated_output_indices() {
//     let a = Tensor::from([[1, 2], [3, 4]]);
//     let result = einsum([&a], (["ij"], "ii"));
//     let expected = Tensor::from([[3, 0], [0, 7]]);
//     assert_almost_eq!(result, expected);
//
//     let result = einsum([&a], (["ii"], "ii"));
//     let expected = Tensor::from([[1, 0], [0, 4]]);
//     assert_almost_eq!(result, expected);
// }
