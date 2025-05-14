use chela::*;

#[test]
#[should_panic]
fn test_einsum_dimension_mismatch() {
    let a = Tensor::from([[1, 2]]);
    let b = Tensor::from([[3, 4], [5, 6], [7, 8]]);
    let _ = einsum([&a, &b], (["ij", "jk"], "ik")); // incompatible shapes
}

#[test]
#[should_panic]
fn test_einsum_invalid_index() {
    let a = Tensor::from([[1, 2]]);
    let b = Tensor::from([[3, 4], [5, 6], [7, 8]]);
    let _ = einsum([&a, &b], (["ij", "kl"], "m")); // invalid index
}

#[test]
fn test_einsum_empty() {
    let _: Tensor<'_, f32> = einsum([], ([], ""));
}

#[test]
fn test_einsum_basic_matmul() {
    let a = Tensor::from([[1, 2], [3, 4]]);
    let b = Tensor::from([[5, 6], [7, 8]]);
    let result = chela::einsum([&a, &b], (["ij", "jk"], "ik"));
    let expected = Tensor::from([[19, 22], [43, 50]]);
    assert_eq!(result, expected);
}

#[test]
fn test_einsum_inner_product() {
    let a = Tensor::from([1, 2, 3]);
    let b = Tensor::from([4, 5, 6]);
    let result = einsum([&a, &b], (["i", "i"], ""));
    let expected = Tensor::scalar(32);  // 1*4 + 2*5 + 3*6
    assert_eq!(result, expected);
}

#[test]
fn test_einsum_outer_product() {
    let a = Tensor::from([1, 2]);
    let b = Tensor::from([3, 4, 5]);
    let result = einsum([&a, &b], (["i", "j"], "ij"));
    let expected = Tensor::from([[3, 4, 5], [6, 8, 10]]);
    assert_eq!(result, expected);
}

#[test]
fn test_einsum_trace() {
    let a = Tensor::from([[1, 2],
                                            [3, 4]]);
    let result = einsum([&a], (["ii"], ""));
    let expected = Tensor::scalar(5); // trace: 1 + 4
    assert_eq!(result, expected);
}

#[test]
fn test_einsum_three_tensors() {
    let a = Tensor::from([[1, 2], [3, 4], [-5, 10]]);
    let b = Tensor::from([[5, 6, 12, 2], [7, 8, -1, 0]]);
    let c = Tensor::from([[1, 0], [0, 1], [0, -1], [-1, 0]]);
    let result = einsum([&a, &b, &c], (["ij", "jk", "kl"], "il"));
    let expected = einsum([&einsum([&a, &b], (["ij", "jk"], "ik")), &c], (["ik", "kl"], "il"));
    assert_eq!(result, expected);
}

#[test]
fn test_einsum_scalar_times_tensor() {
    let a = Tensor::from([[1, 2], [3, 4]]);
    let b = Tensor::scalar(10);
    let result = einsum([&a, &b], (["ij", ""], "ij"));
    let expected = Tensor::from([[10, 20], [30, 40]]);
    assert_eq!(result, expected);
}

#[test]
fn test_einsum_transpose() {
    let a = Tensor::from([[1, 2, 3], [4, 5, 6]]);
    let result = einsum([&a], (["ij"], "ji"));
    let expected = Tensor::from([[1, 4], [2, 5], [3, 6]]);
    assert_eq!(result, expected);
}

#[test]
fn test_einsum_sum_axis() {
    let a = Tensor::from([[1, 2, 3], [4, 5, 6]]);
    let result = einsum([&a], (["ij"], "i")); // Sum along axis j
    let expected = Tensor::from([6, 15]);
    assert_eq!(result, expected);
}

#[test]
fn test_einsum_broadcasting_vector_matrix() {
    let a = Tensor::from([1, 2]); // shape: (2,)
    let b = Tensor::from([[3, 4, 5], [6, 7, 8]]); // shape: (2, 3)
    let result = einsum([&a, &b], (["i", "ij"], "ij"));
    let expected = Tensor::from([[3, 4, 5], [12, 14, 16]]);
    assert_eq!(result, expected);

    let b = Tensor::from([[3, 4], [5, 6]]);
    let result = einsum([&a, &b], (["i", "ij"], "ij"));
    let expected = Tensor::from([[3, 4], [10, 12]]);
    assert_eq!(result, expected);
}

#[test]
fn test_einsum_diagonal_extraction() {
    let a = Tensor::from([[1, 2, 3], [4, 5, 6], [7, 8, 9]]);
    let result = einsum([&a], (["ii"], "i")); // Extract diagonal
    let expected = Tensor::from([1, 5, 9]);
    assert_eq!(result, expected);
}

#[test]
fn test_einsum_tensor_contraction() {
    let a = Tensor::from([[[1, 2], [3, 4]]]); // shape: (1, 2, 2)
    let b = Tensor::from([[5, 6], [7, 8]]);   // shape: (2, 2)
    let result = einsum([&a, &b], (["ijk", "kl"], "ijl"));
    let expected = Tensor::from([[[19, 22], [43, 50]]]);
    assert_eq!(result, expected);
}

#[test]
fn test_einsum_all_sum() {
    let a = Tensor::from([[1, 2], [3, 4]]);
    let result = einsum([&a], (["ij"], ""));
    let expected = Tensor::scalar(10);
    assert_eq!(result, expected);
}

#[test]
fn test_einsum_identity() {
    let a = Tensor::from([[9, 8], [7, 6]]);
    let result = einsum([&a], (["ij"], "ij"));
    assert_eq!(result, a);

    let a = Tensor::from([[0, 0], [0, 0]]);
    let result = einsum([&a], (["ij"], "ij"));
    assert_eq!(result, a);
}

#[test]
fn test_einsum_batch_matmul() {
    let a = Tensor::from([[[1, 2], [3, 4]]]); // shape: (1, 2, 2)
    let b = Tensor::from([[[5, 6], [7, 8]]]); // shape: (1, 2, 2)
    let result = einsum([&a, &b], (["bij", "bjk"], "bik"));
    let expected = Tensor::from([[[19, 22], [43, 50]]]);
    assert_eq!(result, expected);
}

#[test]
fn test_einsum_repeated_output_indices() {
    let a = Tensor::from([[1, 2], [3, 4]]);
    let result = einsum([&a], (["ij"], "ii"));
    let expected = Tensor::from([[3, 0], [0, 7]]);
    assert_eq!(result, expected);

    let result = einsum([&a], (["ii"], "ii"));
    let expected = Tensor::from([[1, 0], [0, 4]]);
    assert_eq!(result, expected);
}
