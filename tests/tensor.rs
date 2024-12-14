use chela::*;

#[test]
fn from_vector() {
    let arr = Tensor::from(vec![0, 50, 100]);
    assert_eq!(arr.len(), &3);
    assert_eq!(arr.shape(), &vec![3]);
    assert_eq!(arr.stride(), &vec![1]);
    assert_eq!(arr.ndims(), 1);
    assert_eq!(arr.size(), 3);

    let arr = Tensor::from(vec![vec![50], vec![50], vec![50]]);
    assert_eq!(arr.len(), &3);
    assert_eq!(arr.shape(), &vec![3, 1]);
    assert_eq!(arr.stride(), &vec![1, 1]);
    assert_eq!(arr.ndims(), 2);
    assert_eq!(arr.size(), 3);

    let arr = Tensor::from(vec![vec![vec![50]], vec![vec![50]]]);
    assert_eq!(arr.len(), &2);
    assert_eq!(arr.shape(), &vec![2, 1, 1]);
    assert_eq!(arr.stride(), &vec![1, 1, 1]);
    assert_eq!(arr.ndims(), 3);
    assert_eq!(arr.size(), 2);

    let arr = Tensor::from(vec![vec![vec![50, 50, 50]], vec![vec![50, 50, 50]]]);
    assert_eq!(arr.len(), &2);
    assert_eq!(arr.shape(), &vec![2, 1, 3]);
    assert_eq!(arr.stride(), &vec![3, 3, 1]);
    assert_eq!(arr.ndims(), 3);
    assert_eq!(arr.size(), 6);
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

#[test]
fn slice_along_nd() {
    let a = Tensor::from([[10], [20], [30], [40]]);

    let slice = a.slice_along(Axis(0), 1);
    assert_eq!(slice.len(), &1);
    assert_eq!(slice.shape(), &vec![1]);
    assert_eq!(slice.ndims(), 1);
    assert_eq!(slice[0], 20);

    let slice = a.slice_along(Axis(1), 0);
    assert_eq!(slice.len(), &4);
    assert_eq!(slice.shape(), &vec![4]);
    assert_eq!(slice.ndims(), 1);
    assert_eq!(slice[0], 10);

    let a = Tensor::from([
        [[10, 20, 30], [40, 50, 60]],
        [[70, 80, 90], [100, 110, 120]],
    ]);

    let slice = a.slice_along(Axis(2), 2);
    assert_eq!(slice.len(), &2);
    assert_eq!(slice.shape(), &vec![2, 2]);
    assert_eq!(slice.ndims(), 2);
    assert_eq!(slice[[0, 0]], 30);
    assert_eq!(slice[[1, 0]], 90);

    let slice = a.slice_along(Axis(1), 1);
    assert_eq!(slice.len(), &2);
    assert_eq!(slice.shape(), &vec![2, 3]);
    assert_eq!(slice.ndims(), 2);
    assert_eq!(slice[[0, 0]], 40);
    assert_eq!(slice[[1, 2]], 120);

    let slice = a.slice_along(Axis(2), 1..);

    assert_eq!(slice.len(), &2);
    assert_eq!(slice.shape(), &vec![2, 2, 2]);
    assert_eq!(slice.ndims(), 3);

    assert_eq!(slice[[0, 0, 0]], 20);
    assert_eq!(slice[[0, 0, 1]], 30);
    assert_eq!(slice[[0, 1, 0]], 50);
    assert_eq!(slice[[1, 0, 0]], 80);
    assert_eq!(slice[[1, 1, 1]], 120);

    let slice = a.slice_along(Axis(1), 1..);

    assert_eq!(slice.len(), &2);
    assert_eq!(slice.shape(), &vec![2, 1, 3]);
    assert_eq!(slice.ndims(), 3);

    assert_eq!(slice[[0, 0, 0]], 40);
    assert_eq!(slice[[0, 0, 2]], 60);
    assert_eq!(slice[[1, 0, 0]], 100);
    assert_eq!(slice[[1, 0, 2]], 120);
}

#[test]
fn slice_homogenous() {
    let a = Tensor::from([
        [[1, 2, 3], [4, 5, 6]],
        [[7, 8, 9], [10, 11, 12]],
    ]);

    let slice = a.slice([1, 1]);

    assert_eq!(slice.len(), &3);
    assert_eq!(slice.shape(), &vec![3]);
    assert_eq!(slice.ndims(), 1);

    assert_eq!(slice[0], 10);
    assert_eq!(slice[1], 11);
    assert_eq!(slice[2], 12);

    let slice = a.slice([1..=1, 1..=1]);

    assert_eq!(slice.len(), &1);
    assert_eq!(slice.shape(), &vec![1, 1, 3]);
    assert_eq!(slice.ndims(), 3);

    assert_eq!(slice[[0, 0, 0]], 10);
    assert_eq!(slice[[0, 0, 1]], 11);
    assert_eq!(slice[[0, 0, 2]], 12);

    let slice = a.slice([0..=0, 0..=1, 0..=1]);

    assert_eq!(slice.len(), &1);
    assert_eq!(slice.shape(), &vec![1, 2, 2]);
    assert_eq!(slice.ndims(), 3);

    assert_eq!(slice[[0, 0, 0]], 1);
    assert_eq!(slice[[0, 0, 1]], 2);
    assert_eq!(slice[[0, 1, 0]], 4);
    assert_eq!(slice[[0, 1, 1]], 5);
}

#[test]
fn slice_heterogeneous() {
    let a = Tensor::from([
        [[1, 2, 3], [4, 5, 6]],
        [[7, 8, 9], [10, 11, 12]],
        [[13, 14, 15], [16, 17, 18]],
    ]);

    let slice = a.slice(s![0, .., 0..=1]);

    assert_eq!(slice.len(), &2);
    assert_eq!(slice.shape(), &vec![2, 2]);
    assert_eq!(slice.ndims(), 2);
}

#[test]
fn flat_iter() {
    let a = Tensor::from([
        [[10, 11, 12], [13, 14, 15]],
        [[16, 17, 18], [19, 20, 21]],
        [[22, 23, 24], [25, 26, 27]],
    ]);

    let slice: Vec<_> = a.flat_iter().collect();
    assert_eq!(slice, [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27]);

    let b = a.slice(s![.., 0]);
    let slice: Vec<_> = b.flat_iter().collect();
    assert_eq!(slice, [10, 11, 12, 16, 17, 18, 22, 23, 24]);

    let b = a.slice(s![1]);
    println!("{:?}", b);
    let slice: Vec<_> = b.flat_iter().collect();
    assert_eq!(slice, [16, 17, 18, 19, 20, 21]);

    let b = a.slice(s![..2, 1, 1..]);
    let slice: Vec<_> = b.flat_iter().collect();
    assert_eq!(slice, [14, 15, 20, 21]);
}
