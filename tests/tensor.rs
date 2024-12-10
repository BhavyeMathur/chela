use chela::tensor::*;

#[test]
fn from_vector() {
    let arr = Tensor::from(vec![0, 50, 100]);
    assert_eq!(arr.len(), &3);
    assert_eq!(arr.shape(), &vec![3]);

    let arr = Tensor::from(vec![vec![50], vec![50], vec![50]]);
    assert_eq!(arr.len(), &3);
    assert_eq!(arr.shape(), &vec![3, 1]);

    let arr = Tensor::from(vec![vec![vec![50]], vec![vec![50]]]);
    assert_eq!(arr.len(), &2);
    assert_eq!(arr.shape(), &vec![2, 1, 1]);

    let arr = Tensor::from(vec![vec![vec![50, 50, 50]], vec![vec![50, 50, 50]]]);
    assert_eq!(arr.len(), &2);
    assert_eq!(arr.shape(), &vec![2, 1, 3]);
}

#[test]
fn from_array() {
    let arr = Tensor::from([500, 50, 100]);
    assert_eq!(arr.len(), &3);
    assert_eq!(arr.shape(), &vec![3]);

    let arr = Tensor::from([[500], [50], [100]]);
    assert_eq!(arr.len(), &3);
    assert_eq!(arr.shape(), &vec![3, 1]);

    let arr = Tensor::from([[[500], [50], [30]], [[50], [0], [0]]]);
    assert_eq!(arr.len(), &2);
    assert_eq!(arr.shape(), &vec![2, 3, 1]);

    let arr = Tensor::from([[[50, 50, 50]], [[50, 50, 50]]]);
    assert_eq!(arr.len(), &2);
    assert_eq!(arr.shape(), &vec![2, 1, 3]);
}

#[test]
#[should_panic]
fn from_inhomogeneous_vector1() {
    Tensor::from(vec![vec![50, 50], vec![50]]);
}

#[test]
#[should_panic]
fn from_inhomogeneous_vector2() {
    Tensor::from(vec![vec![vec![50, 50]], vec![vec![50]], vec![vec![50]]]);
}
