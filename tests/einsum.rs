use chela::*;

#[test]
#[should_panic]
fn test_einsum_dimension_mismatch() {
    let a = Tensor::from([[1, 2]]);
    let b = Tensor::from([[3, 4], [5, 6], [7, 8]]);
    let _ = einsum(&[&a, &b], (["ij", "jk"], "ik")); // incompatible shapes
}

#[test]
#[should_panic]
fn test_einsum_invalid_index() {
    let a = Tensor::from([[1, 2]]);
    let b = Tensor::from([[3, 4], [5, 6], [7, 8]]);
    let _ = einsum(&[&a, &b], (["ij", "kl"], "m")); // invalid index
}

#[test]
fn test_einsum_empty() {
    let _: Tensor<'_, f32> = einsum(&[], ([], ""));
}

#[test]
fn test_einsum_basic_matmul() {
    let a = Tensor::from([[1, 2], [3, 4]]);
    let b = Tensor::from([[5, 6], [7, 8]]);
    let result = chela::einsum(&[&a, &b], (["ij", "jk"], "ik"));
    let expected = Tensor::from([[19, 22], [43, 50]]);
    assert_eq!(result, expected);
}

#[test]
fn test_einsum_pointwise_multiplication() {
    let a = Tensor::from([[1, 2, 3], [0, 1, 2], [4, 5, 6]]);
    let b = Tensor::from([[5, 6, 7], [10, 20, 30], [3, 6, 9]]);
    let result = chela::einsum(&[&a, &b], (["ij", "ij"], "ij"));
    let expected = Tensor::from([[5, 12, 21], [0, 20, 60], [12, 30, 54]]);
    assert_eq!(result, expected);
}

#[test]
fn test_einsum_sum_along_axes() {
    let a = Tensor::from([[1f32, 2.0], [0.0, 1.0]]);
    let b = Tensor::from([[5.0, 6.0], [10.0, 20.0]]);

    let expected = Tensor::from([71f32, 30.0]);
    let result = chela::einsum(&[&a, &b], (["ij", "jk"], "i"));
    assert_eq!(result, expected);

    let expected = Tensor::from([11f32, 90.0]);
    let result = chela::einsum(&[&a, &b], (["ij", "jk"], "j"));
    assert_eq!(result, expected);
}

#[test]
fn test_einsum_sum_product() {
    let a = Tensor::from([[1f64, 2.0], [0.0, 1.0]]);
    let b = Tensor::from([[5.0, 6.0], [10.0, 20.0]]);

    let expected = Tensor::scalar(63.0);
    let result = chela::einsum(&[&a, &b], (["ij", "ik"], ""));
    assert_eq!(result, expected);

    let expected = Tensor::scalar(101.0);
    let result = chela::einsum(&[&a, &b], (["ij", "jk"], ""));
    assert_eq!(result, expected);

    let expected = Tensor::scalar(71.0);
    let result = chela::einsum(&[&a, &b], (["ij", "ki"], ""));
    assert_eq!(result, expected);

    let expected = Tensor::scalar(93.0);
    let result = chela::einsum(&[&a, &b], (["ij", "kj"], ""));
    assert_eq!(result, expected);
}

#[test]
fn test_einsum_2operands_to_3d() {
    let a = Tensor::from([[1u8, 2], [0, 1]]);
    let b = Tensor::from([[5, 6], [10, 20]]);

    let expected = Tensor::from([[[5u8, 10], [12, 40]], [[0, 0], [6, 20]]]);
    let result = chela::einsum(&[&a, &b], (["ij", "kj"], "ijk"));
    assert_eq!(result, expected);

    let expected = Tensor::from([[[5u8, 6], [10, 12]], [[0, 0], [10, 20]]]);
    let result = chela::einsum(&[&a, &b], (["ij", "ik"], "ijk"));
    assert_eq!(result, expected);
}

#[test]
fn test_einsum_inner_product() {
    let a = Tensor::from([1, 2, 3]);
    let b = Tensor::from([4, 5, 6]);
    let result = einsum(&[&a, &b], (["i", "i"], ""));
    let expected = Tensor::scalar(32);  // 1*4 + 2*5 + 3*6
    assert_eq!(result, expected);
}

#[test]
fn test_einsum_outer_product() {
    let a = Tensor::from([1, 2]);
    let b = Tensor::from([3, 4, 5]);
    let result = einsum(&[&a, &b], (["i", "j"], "ij"));
    let expected = Tensor::from([[3, 4, 5], [6, 8, 10]]);
    assert_eq!(result, expected);
}

#[test]
fn test_einsum_matrix_outer_product() {
    let a = Tensor::from([[1, 2], [3, 4]]);
    let b = Tensor::from([[5, 6], [7, 8]]);
    let result = chela::einsum(&[&a, &b], (["ij", "kl"], "ijkl"));
    let expected = Tensor::from([
        [[[5, 6], [7, 8]], [[10, 12], [14, 16]]],
        [[[15, 18], [21, 24]], [[20, 24], [28, 32]]]
    ]);
    assert_eq!(result, expected);
}

#[test]
fn test_einsum_trace() {
    let a = Tensor::from([[1, 2],
        [3, 4]]);
    let result = einsum(&[&a], (["ii"], ""));
    let expected = Tensor::scalar(5); // trace: 1 + 4
    assert_eq!(result, expected);
}

#[test]
fn test_einsum_three_tensors() {
    let a = Tensor::from([[1, 2], [3, 4], [-5, 10]]);
    let b = Tensor::from([[5, 6, 12, 2], [7, 8, -1, 0]]);
    let c = Tensor::from([[1, 0], [0, 1], [0, -1], [-1, 0]]);
    let result = einsum(&[&a, &b, &c], (["ij", "jk", "kl"], "il"));
    let expected = einsum(&[&einsum(&[&a, &b], (["ij", "jk"], "ik")), &c], (["ik", "kl"], "il"));
    assert_eq!(result, expected);
}

#[test]
fn test_einsum_scalar_times_tensor() {
    let a = Tensor::from([[1, 2], [3, 4]]);
    let b = Tensor::scalar(10);
    let result = einsum(&[&a, &b], (["ij", ""], "ij"));
    let expected = Tensor::from([[10, 20], [30, 40]]);
    assert_eq!(result, expected);
}

#[test]
fn test_einsum_transpose() {
    let a = Tensor::from([[1, 2, 3], [4, 5, 6]]);
    let expected = Tensor::from([[1, 4], [2, 5], [3, 6]]);

    let result = einsum(&[&a], (["ij"], "ji"));
    assert_eq!(result, expected);

    let result = einsum_view(&a, ("ij", "ji")).unwrap();
    assert_eq!(result, expected);
}

#[test]
fn test_einsum_sum_axis() {
    let a = Tensor::from([[1, 2, 3], [4, 5, 6]]);
    let result = einsum(&[&a], (["ij"], "i")); // Sum along axis j
    let expected = Tensor::from([6, 15]);
    assert_eq!(result, expected);
}

#[test]
fn test_einsum_broadcasting_vector_matrix() {
    let a = Tensor::from([1, 2]); // shape: (2,)
    let b = Tensor::from([[3, 4, 5], [6, 7, 8]]); // shape: (2, 3)
    let result = einsum(&[&a, &b], (["i", "ij"], "ij"));
    let expected = Tensor::from([[3, 4, 5], [12, 14, 16]]);
    assert_eq!(result, expected);

    let b = Tensor::from([[3, 4], [5, 6]]);
    let result = einsum(&[&a, &b], (["i", "ij"], "ij"));
    let expected = Tensor::from([[3, 4], [10, 12]]);
    assert_eq!(result, expected);
}

#[test]
fn test_einsum_diagonal_extraction() {
    let a = Tensor::from([[1, 2, 3], [4, 5, 6], [7, 8, 9]]);
    let expected = Tensor::from([1, 5, 9]);

    let result = einsum(&[&a], (["ii"], "i"));
    assert_eq!(result, expected);

    let result = einsum_view(&a, ("ii", "i")).unwrap();
    assert_eq!(result, expected);
}

#[test]
fn test_einsum_tensor_contraction() {
    let a = Tensor::from([[[1, 2], [3, 4]]]); // shape: (1, 2, 2)
    let b = Tensor::from([[5, 6], [7, 8]]);   // shape: (2, 2)
    let result = einsum(&[&a, &b], (["ijk", "kl"], "ijl"));
    let expected = Tensor::from([[[19, 22], [43, 50]]]);
    assert_eq!(result, expected);
}

#[test]
fn test_einsum_all_sum() {
    let a = Tensor::from([[1, 2], [3, 4]]);
    let result = einsum(&[&a], (["ij"], ""));
    let expected = Tensor::scalar(10);
    assert_eq!(result, expected);
}

#[test]
fn test_einsum_identity() {
    let a = Tensor::from([[9, 8], [7, 6]]);
    let result = einsum(&[&a], (["ij"], "ij"));
    assert_eq!(result, a);

    let a = Tensor::from([[0, 0], [0, 0]]);
    let result = einsum(&[&a], (["ij"], "ij"));
    assert_eq!(result, a);

    let result = einsum_view(&a, ("ij", "ij")).unwrap();
    assert_eq!(result, a);
}

#[test]
fn test_einsum_batch_matmul() {
    let a = Tensor::from([[[1, 2], [3, 4]]]); // shape: (1, 2, 2)
    let b = Tensor::from([[[5, 6], [7, 8]]]); // shape: (1, 2, 2)
    let result = einsum(&[&a, &b], (["bij", "bjk"], "bik"));
    let expected = Tensor::from([[[19, 22], [43, 50]]]);
    assert_eq!(result, expected);
}

// #[test]
// fn test_einsum_repeated_output_indices() {
//     let a = Tensor::from([[1, 2], [3, 4]]);
//     let result = einsum(&[&a], (["ij"], "ii"));
//     let expected = Tensor::from([[3, 0], [0, 7]]);
//     assert_eq!(result, expected);
//
//     let result = einsum(&[&a], (["ii"], "ii"));
//     let expected = Tensor::from([[1, 0], [0, 4]]);
//     assert_eq!(result, expected);
// }
