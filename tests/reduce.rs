use chela::*;

#[test]
fn test_reduce_sum() {
    let tensor = Tensor::from([[1, 1], [2, 2], [3, 3]]);

    let correct = Tensor::from([2, 4, 6]);
    let output: Tensor<i32> = tensor.sum(0);
    assert_eq!(output, correct);

    let output: Tensor<i32> = tensor.sum(Axis(0));
    assert_eq!(output, correct);

    let correct = tensor.clone();
    let output: Tensor<i32> = tensor.sum([0, 1]);
    assert_eq!(output, correct);

    let correct = Tensor::scalar(12);
    let output: Tensor<i32> = tensor.sum([]);
    assert_eq!(output, correct);
}

#[test]
fn test_reduce_multiply() {
    let tensor = Tensor::from([[1, 1], [2, 2], [3, 3]]);

    let correct = Tensor::from([1, 4, 9]);
    let output: Tensor<i32> = tensor.product(0);
    assert_eq!(output, correct);

    let correct = tensor.clone();
    let output: Tensor<i32> = tensor.product([0, 1]);
    assert_eq!(output, correct);

    let correct = Tensor::scalar(36);
    let output: Tensor<i32> = tensor.product([]);
    assert_eq!(output, correct);
}
