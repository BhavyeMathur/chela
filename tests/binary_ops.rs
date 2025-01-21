use chela::*;

#[test]
#[should_panic]
fn test_broadcast_panic() {
    let tensor1 = Tensor::from([[1, 2], [3, 4], [5, 6]]);
    let tensor2 = Tensor::from([2, 4, 6]);
    let _ = tensor1 + tensor2;
}

#[test]
fn test_add() {
    let tensor1 = Tensor::from([[1, 1], [2, 2], [3, 3]]);
    let tensor2 = Tensor::from([[1, 2], [3, 4], [5, 6]]);

    let correct = Tensor::from([[2, 3], [5, 6], [8, 9]]);
    let output = tensor1 + tensor2;

    assert_eq!(output, correct);
}

#[test]
fn test_broadcast_add() {
    let tensor1 = Tensor::from([[1, 2], [3, 4], [5, 6]]);
    let tensor2 = Tensor::from([2, 4]);

    let correct = Tensor::from([[3, 6], [5, 8], [7, 10]]);
    let output = tensor1 + tensor2;

    assert_eq!(output, correct);
}

#[test]
fn test_scalar_add1() {
    let tensor1 = Tensor::from([[1, 2], [3, 4], [5, 6]]);

    let correct = Tensor::from([[6, 7], [8, 9], [10, 11]]);
    let output = tensor1 + 5;

    assert_eq!(output, correct);
}

#[test]
fn test_add_lifetimes() {
    let output;

    {
        let tensor1 = Tensor::from([[1, 2], [3, 4], [5, 6]]);
        let tensor2 = Tensor::from([2, 4]);
        output = tensor1 + tensor2;
    }

    let correct = Tensor::from([[3, 6], [5, 8], [7, 10]]);
    assert_eq!(output, correct);
}

#[test]
fn test_scalar_add_lifetimes() {
    let output;

    {
        let tensor1 = Tensor::from([[1, 2], [3, 4], [5, 6]]);
        output = tensor1 + 5;
    }

    let correct = Tensor::from([[6, 7], [8, 9], [10, 11]]);
    assert_eq!(output, correct);
}

#[test]
fn test_add_reference() {
    let tensor1 = Tensor::from([[1, 1], [2, 2], [3, 3]]);
    let tensor2 = Tensor::from([[1, 2], [3, 4], [5, 6]]);

    let correct = Tensor::from([[2, 3], [5, 6], [8, 9]]);
    let output = tensor1 + &tensor2;

    assert_eq!(tensor2, tensor2);
    assert_eq!(output, correct);
}

#[test]
fn test_scalar_add_reference() {
    let tensor1 = Tensor::from([[1, 1], [2, 2], [3, 3]]);
    let tensor2 = Tensor::from([[1, 2], [3, 4], [5, 6]]);

    let correct = Tensor::from([[2, 3], [5, 6], [8, 9]]);
    let output = tensor1 + &tensor2;

    assert_eq!(tensor2, tensor2);
    assert_eq!(output, correct);
}

#[test]
fn test_add_references() {
    let tensor1 = Tensor::from([[1, 2], [3, 4], [5, 6]]);

    let correct = Tensor::from([[6, 7], [8, 9], [10, 11]]);
    let output = &tensor1 + 5;

    assert_eq!(tensor1, tensor1);
    assert_eq!(output, correct);
}
