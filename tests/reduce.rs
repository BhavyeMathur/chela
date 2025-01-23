use chela::*;

#[test]
#[should_panic]
fn test_reduce_panic() {
    let tensor = Tensor::from([[1, 1], [2, 2], [3, 3]]);
    tensor.sum_along([0, 0]);
}

#[test]
fn test_reduce_sum() {
    let tensor = Tensor::from([[1, 1], [2, 2], [3, 3]]);

    let correct = Tensor::from([2, 4, 6]);
    let output = tensor.sum_along(0);
    assert_eq!(output, correct);

    let output = tensor.sum_along(Axis(0));
    assert_eq!(output, correct);

    let correct = tensor.clone();
    let output = tensor.sum_along([0, 1]);
    assert_eq!(output, correct);

    let correct = Tensor::scalar(12);
    let output = tensor.sum();
    assert_eq!(output, correct);
}

#[test]
fn test_reduce_multiply() {
    let tensor = Tensor::from([[1, 1], [2, 2], [3, 3]]);

    let correct = Tensor::from([1, 4, 9]);
    let output = tensor.product_along(0);
    assert_eq!(output, correct);

    let correct = tensor.clone();
    let output = tensor.product_along([0, 1]);
    assert_eq!(output, correct);

    let correct = Tensor::scalar(36);
    let output = tensor.product();
    assert_eq!(output, correct);
}

#[test]
fn test_reduce_mean() {
    let tensor = Tensor::from([[1, 3], [2, 4], [3, 5]]);

    let correct = Tensor::from([2.0f32, 3.0, 4.0]);
    let output = tensor.mean_along(0);
    assert_eq!(output, correct);

    let correct = Tensor::from([[1.0f32, 3.0], [2.0, 4.0], [3.0, 5.0]]);
    let output = tensor.mean_along([0, 1]);
    assert_eq!(output, correct);

    let correct = Tensor::scalar(3.0f32);
    let output = tensor.mean();
    assert_eq!(output, correct);
}

#[test]
fn test_reduce_min() {
    let tensor = Tensor::from([[1, 3], [2, 4], [3, 5]]);

    let correct = Tensor::from([1, 2, 3]);
    let output = tensor.min_along(0);
    assert_eq!(output, correct);

    let correct = Tensor::scalar(1);
    let output = tensor.min();
    assert_eq!(output, correct);
}

#[test]
fn test_reduce_max() {
    let tensor = Tensor::from([[1, 3], [2, 4], [3, 5]]);

    let correct = Tensor::from([3, 4, 5]);
    let output = tensor.max_along(0);
    assert_eq!(output, correct);

    let correct = Tensor::scalar(5);
    let output = tensor.max();
    assert_eq!(output, correct);
}

#[test]
fn test_reduce_min_f32() {
    let tensor = Tensor::from([[1.0f32, 3.0], [2.0, 4.0], [3.0, 5.0]]);

    let correct = Tensor::from([1.0f32, 2.0, 3.0]);
    let output = tensor.min_along(0);
    assert_eq!(output, correct);

    let correct = Tensor::scalar(1.0f32);
    let output = tensor.min();
    assert_eq!(output, correct);
}

#[test]
fn test_reduce_max_f64() {
    let tensor = Tensor::from([[1.0f64, 3.0], [2.0, 4.0], [3.0, 5.0]]);

    let correct = Tensor::from([3.0f64, 4.0, 5.0]);
    let output = tensor.max_along(0);
    assert_eq!(output, correct);

    let correct = Tensor::scalar(5.0f64);
    let output = tensor.max();
    assert_eq!(output, correct);
}