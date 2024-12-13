use chela::*;

#[test]
fn from_vector() {
    let arr = Tensor::from(vec![0, 50, 100]);
    assert_eq!(arr.len(), &3);
    assert_eq!(arr.shape(), &vec![3]);
    assert_eq!(arr.stride(), &vec![1]);
    assert_eq!(arr.ndims(), 1);

    let arr = Tensor::from(vec![vec![50], vec![50], vec![50]]);
    assert_eq!(arr.len(), &3);
    assert_eq!(arr.shape(), &vec![3, 1]);
    assert_eq!(arr.stride(), &vec![1, 1]);
    assert_eq!(arr.ndims(), 2);

    let arr = Tensor::from(vec![vec![vec![50]], vec![vec![50]]]);
    assert_eq!(arr.len(), &2);
    assert_eq!(arr.shape(), &vec![2, 1, 1]);
    assert_eq!(arr.stride(), &vec![1, 1, 1]);
    assert_eq!(arr.ndims(), 3);

    let arr = Tensor::from(vec![vec![vec![50, 50, 50]], vec![vec![50, 50, 50]]]);
    assert_eq!(arr.len(), &2);
    assert_eq!(arr.shape(), &vec![2, 1, 3]);
    assert_eq!(arr.stride(), &vec![3, 3, 1]);
    assert_eq!(arr.ndims(), 3);
}

#[test]
fn from_array() {
    let arr = Tensor::from([500, 50, 100]);
    assert_eq!(arr.len(), &3);
    assert_eq!(arr.shape(), &vec![3]);
    assert_eq!(arr.stride(), &vec![1]);
    assert_eq!(arr.ndims(), 1);

    let arr = Tensor::from([[500], [50], [100]]);
    assert_eq!(arr.len(), &3);
    assert_eq!(arr.shape(), &vec![3, 1]);
    assert_eq!(arr.stride(), &vec![1, 1]);
    assert_eq!(arr.ndims(), 2);

    let arr = Tensor::from([[[500], [50], [30]], [[50], [0], [0]]]);
    assert_eq!(arr.len(), &2);
    assert_eq!(arr.shape(), &vec![2, 3, 1]);
    assert_eq!(arr.stride(), &vec![3, 1, 1]);
    assert_eq!(arr.ndims(), 3);

    let arr = Tensor::from([[[50, 50, 50]], [[50, 50, 50]]]);
    assert_eq!(arr.len(), &2);
    assert_eq!(arr.shape(), &vec![2, 1, 3]);
    assert_eq!(arr.stride(), &vec![3, 3, 1]);
    assert_eq!(arr.ndims(), 3);
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

#[test]
fn println() {
    println!("{:?}", Tensor::from([[[10, 20], [30, 40]]]));
    println!("{:?}", Tensor::from([vec![vec![5, 10], vec![500, 100]]]));
}

#[test]
fn index() {
    let a = Tensor::from([10, 20, 30, 40]);
    assert_eq!(a[0], 10);
    assert_eq!(a[3], 40);

    let a = Tensor::from([[10, 20], [30, 40]]);
    assert_eq!(a[[0, 1]], 20);
    assert_eq!(a[[1, 1]], 40);
}

#[test]
fn slice_along_1d() {
    let a = Tensor::from([10, 20, 30, 40]);

    let slice = a.slice_along(Axis(0), 1);
    assert_eq!(slice.len(), &0);
    assert_eq!(slice.shape(), &vec![]);
    assert_eq!(slice.ndims(), 0);

    let slice = a.slice_along(Axis(0), ..);
    assert_eq!(slice.len(), &4);
    assert_eq!(slice[0], 10);
    assert_eq!(slice[3], 40);
    assert_eq!(slice.shape(), &vec![4]);
    assert_eq!(slice.ndims(), 1);

    let slice = a.slice_along(Axis(0), 2..);
    assert_eq!(slice.len(), &2);
    assert_eq!(slice[0], 30);
    assert_eq!(slice[1], 40);
    assert_eq!(slice.shape(), &vec![2]);
    assert_eq!(slice.ndims(), 1);

    let slice = a.slice_along(Axis(0), ..3);
    assert_eq!(slice.len(), &3);
    assert_eq!(slice[0], 10);
    assert_eq!(slice[2], 30);
    assert_eq!(slice.shape(), &vec![3]);
    assert_eq!(slice.ndims(), 1);

    let slice = a.slice_along(Axis(0), ..=3);
    assert_eq!(slice.len(), &4);
    assert_eq!(slice[0], 10);
    assert_eq!(slice[3], 40);
    assert_eq!(slice.shape(), &vec![4]);
    assert_eq!(slice.ndims(), 1);

    let slice = a.slice_along(Axis(0), 1..3);
    assert_eq!(slice.len(), &2);
    assert_eq!(slice[0], 20);
    assert_eq!(slice[1], 30);
    assert_eq!(slice.shape(), &vec![2]);
    assert_eq!(slice.ndims(), 1);

    let slice = a.slice_along(Axis(0), 1..=3);
    assert_eq!(slice.len(), &3);
    assert_eq!(slice[0], 20);
    assert_eq!(slice[2], 40);
    assert_eq!(slice.shape(), &vec![3]);
    assert_eq!(slice.ndims(), 1);
}
