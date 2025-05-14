use chela::*;

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
