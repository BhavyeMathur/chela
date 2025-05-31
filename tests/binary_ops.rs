use chela::*;

#[test]
fn test_add() {
    let tensor1 = NdArray::new([[1, 1], [2, 2], [3, 3]]);
    let tensor2 = NdArray::new([[1, 2], [3, 4], [5, 6]]);

    let correct = NdArray::new([[2, 3], [5, 6], [8, 9]]);
    let output = tensor1 + tensor2;

    assert_eq!(output, correct);
}

#[test]
fn test_broadcast_add() {
    let tensor1 = NdArray::new([[1, 2], [3, 4], [5, 6]]);
    let tensor2 = NdArray::new([2, 4]);

    let correct = NdArray::new([[3, 6], [5, 8], [7, 10]]);
    let output = tensor1 + tensor2;

    assert_eq!(output, correct);
}

#[test]
fn test_scalar_add1() {
    let tensor1 = NdArray::new([[1, 2], [3, 4], [5, 6]]);

    let correct = NdArray::new([[6, 7], [8, 9], [10, 11]]);
    let output = tensor1 + 5;

    assert_eq!(output, correct);
}

#[test]
fn test_add_lifetimes() {
    let output;

    {
        let tensor1 = NdArray::new([[1, 2], [3, 4], [5, 6]]);
        let tensor2 = NdArray::new([2, 4]);
        output = tensor1 + tensor2;
    }

    let correct = NdArray::new([[3, 6], [5, 8], [7, 10]]);
    assert_eq!(output, correct);
}

#[test]
fn test_scalar_add_lifetimes() {
    let output;

    {
        let tensor1 = NdArray::new([[1, 2], [3, 4], [5, 6]]);
        output = tensor1 + 5;
    }

    let correct = NdArray::new([[6, 7], [8, 9], [10, 11]]);
    assert_eq!(output, correct);
}

#[test]
fn test_add_reference() {
    let tensor1 = NdArray::new([[1, 1], [2, 2], [3, 3]]);
    let tensor2 = NdArray::new([[1, 2], [3, 4], [5, 6]]);

    let correct = NdArray::new([[2, 3], [5, 6], [8, 9]]);
    let output = tensor1 + &tensor2;

    assert_eq!(tensor2, tensor2);
    assert_eq!(output, correct);
}

#[test]
fn test_scalar_add_reference() {
    let tensor1 = NdArray::new([[1, 1], [2, 2], [3, 3]]);
    let tensor2 = NdArray::new([[1, 2], [3, 4], [5, 6]]);

    let correct = NdArray::new([[2, 3], [5, 6], [8, 9]]);
    let output = tensor1 + &tensor2;

    assert_eq!(tensor2, tensor2);
    assert_eq!(output, correct);
}

#[test]
fn test_add_references() {
    let tensor1 = NdArray::new([[1, 2], [3, 4], [5, 6]]);

    let correct = NdArray::new([[6, 7], [8, 9], [10, 11]]);
    let output = &tensor1 + 5;

    assert_eq!(tensor1, tensor1);
    assert_eq!(output, correct);
}

#[test]
fn test_subtract() {
    let tensor1 = NdArray::new([[1, 1], [2, 2], [3, 3]]);
    let tensor2 = NdArray::new([2, 4]);

    let correct = NdArray::new([[-1, -3], [0, -2], [1, -1]]);
    let output = tensor1 - tensor2;

    assert_eq!(output, correct);
}

#[test]
fn test_multiply() {
    let tensor1 = NdArray::new([[1, 1], [2, 2], [3, 3]]);
    let tensor2 = NdArray::new([2, 4]);

    let correct = NdArray::new([[2, 4], [4, 8], [6, 12]]);
    let output = tensor1 * tensor2;

    assert_eq!(output, correct);
}

#[test]
fn test_divide() {
    let tensor1 = NdArray::new([[1.0, 1.0], [2.0, 2.0], [6.0, 8.0]]);
    let tensor2 = NdArray::new([2.0, 4.0]);

    let correct = NdArray::new([[0.5, 0.25], [1.0, 0.5], [3.0, 2.0]]);
    assert_eq!(&tensor1 / &tensor2, correct);
    assert_eq!(&tensor1 / tensor2, correct);

    let tensor2 = NdArray::new([2.0, 4.0]);
    assert_eq!(tensor1 / &tensor2, correct);

    let tensor1 = NdArray::new([[1.0, 1.0], [2.0, 2.0], [6.0, 8.0]]);
    assert_eq!(tensor1 / tensor2, correct);
}

#[test]
fn test_remainder() {
    let tensor1 = NdArray::new([[1, 3], [2, 2], [3, 7]]);
    let tensor2 = NdArray::new([2, 4]);

    let correct = NdArray::new([[1, 3], [0, 2], [1, 3]]);
    let output = tensor1 % tensor2;

    assert_eq!(output, correct);
}

#[test]
fn test_bitand() {
    let tensor1 = NdArray::new([[1, 3], [2, 2], [3, 7]]);
    let tensor2 = NdArray::new([1, 2]);

    let correct = NdArray::new([[1, 2], [0, 2], [1, 2]]);
    let output = tensor1 & tensor2;

    assert_eq!(output, correct);
}

#[test]
fn test_bitor() {
    let tensor1 = NdArray::new([[1, 3], [2, 2], [3, 7]]);
    let tensor2 = NdArray::new([1, 2]);

    let correct = NdArray::new([[1, 3], [3, 2], [3, 7]]);
    let output = tensor1 | tensor2;

    assert_eq!(output, correct);
}

#[test]
fn test_shl() {
    let tensor1 = NdArray::new([[1, 2], [4, 8], [16, 32]]);
    let tensor2 = NdArray::new([1, 2]);

    let correct = NdArray::new([[2, 8], [8, 32], [32, 128]]);
    let output = tensor1 << tensor2;

    assert_eq!(output, correct);
}

#[test]
fn test_shr() {
    let tensor1 = NdArray::new([[2, 4], [8, 16], [32, 64]]);
    let tensor2 = NdArray::new([1, 2]);

    let correct = NdArray::new([[1, 1], [4, 4], [16, 16]]);
    let output = tensor1 >> tensor2;

    assert_eq!(output, correct);
}

#[test]
fn test_iadd() {
    let mut tensor1 = NdArray::new([[1, 2], [3, 4], [5, 6]]);
    let tensor2 = NdArray::new([2, 4]);

    let correct = NdArray::new([[3, 6], [5, 8], [7, 10]]);
    tensor1 += &tensor2;
    assert_eq!(tensor1, correct);

    let correct = NdArray::new([[5, 10], [7, 12], [9, 14]]);
    tensor1 += tensor2;
    assert_eq!(tensor1, correct);

    let correct = NdArray::new([[10, 15], [12, 17], [14, 19]]);
    tensor1 += 5;
    assert_eq!(tensor1, correct);
}

#[test]
fn test_isub() {
    let mut tensor1 = NdArray::new([[1, 2], [3, 4], [5, 6]]);
    let tensor2 = NdArray::new([2, 4]);

    let correct = NdArray::new([[-1, -2], [1, 0], [3, 2]]);
    tensor1 -= &tensor2;
    assert_eq!(tensor1, correct);

    let correct = NdArray::new([[-3, -6], [-1, -4], [1, -2]]);
    tensor1 -= tensor2;
    assert_eq!(tensor1, correct);

    let correct = NdArray::new([[-4, -7], [-2, -5], [0, -3]]);
    tensor1 -= 1;
    assert_eq!(tensor1, correct);
}

#[test]
#[should_panic]
fn test_broadcast_panic() {
    let tensor1 = NdArray::new([[1, 2], [3, 4], [5, 6]]);
    let tensor2 = NdArray::new([2, 4, 6]);
    let _ = tensor1 + tensor2;
}

#[test]
#[should_panic]
fn test_iadd_panic() {
    let mut tensor1 = NdArray::new([2, 4, 6]);
    let tensor2 = NdArray::new([[1, 2], [3, 4], [5, 6]]);
    tensor1 += &tensor2;
}

#[test]
#[should_panic]
fn test_viewonly_panic() {
    let tensor1 = NdArray::new([2, 4, 6]);
    let mut tensor1 = tensor1.broadcast_to(&[3, 3]);

    let tensor2 = NdArray::new([[1, 2, 3], [3, 4, 5], [5, 6, 7]]);
    tensor1 += &tensor2;
}
