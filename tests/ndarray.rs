use redstone::*;

#[test]
fn from_vector() {
    let arr = NdArray::new(vec![0, 50, 100]);
    assert_eq!(arr.len(), 3);
    assert_eq!(arr.shape(), &[3]);
    assert_eq!(arr.stride(), &[1]);
    assert_eq!(arr.ndims(), 1);
    assert_eq!(arr.size(), 3);

    let arr = NdArray::new(vec![vec![50], vec![50], vec![50]]);
    assert_eq!(arr.len(), 3);
    assert_eq!(arr.shape(), &[3, 1]);
    assert_eq!(arr.stride(), &[1, 1]);
    assert_eq!(arr.ndims(), 2);
    assert_eq!(arr.size(), 3);

    let arr = NdArray::new(vec![vec![vec![50]], vec![vec![50]]]);
    assert_eq!(arr.len(), 2);
    assert_eq!(arr.shape(), &[2, 1, 1]);
    assert_eq!(arr.stride(), &[1, 1, 1]);
    assert_eq!(arr.ndims(), 3);
    assert_eq!(arr.size(), 2);

    let arr = NdArray::new(vec![vec![vec![50, 50, 50]], vec![vec![50, 50, 50]]]);
    assert_eq!(arr.len(), 2);
    assert_eq!(arr.shape(), &[2, 1, 3]);
    assert_eq!(arr.stride(), &[3, 3, 1]);
    assert_eq!(arr.ndims(), 3);
    assert_eq!(arr.size(), 6);
}

#[test]
fn from_array() {
    let arr = NdArray::new([500, 50, 100]);
    assert_eq!(arr.len(), 3);
    assert_eq!(arr.shape(), &[3]);
    assert_eq!(arr.stride(), &[1]);
    assert_eq!(arr.ndims(), 1);

    let arr = NdArray::new([[500], [50], [100]]);
    assert_eq!(arr.len(), 3);
    assert_eq!(arr.shape(), &[3, 1]);
    assert_eq!(arr.stride(), &[1, 1]);
    assert_eq!(arr.ndims(), 2);

    let arr = NdArray::new([[[500], [50], [30]], [[50], [0], [0]]]);
    assert_eq!(arr.len(), 2);
    assert_eq!(arr.shape(), &[2, 3, 1]);
    assert_eq!(arr.stride(), &[3, 1, 1]);
    assert_eq!(arr.ndims(), 3);

    let arr = NdArray::new([[[50, 50, 50]], [[50, 50, 50]]]);
    assert_eq!(arr.len(), 2);
    assert_eq!(arr.shape(), &[2, 1, 3]);
    assert_eq!(arr.stride(), &[3, 3, 1]);
    assert_eq!(arr.ndims(), 3);
}

#[test]
#[should_panic]
fn from_inhomogeneous_vector1() {
    NdArray::new(vec![vec![50, 50], vec![50]]);
}

#[test]
#[should_panic]
fn from_inhomogeneous_vector2() {
    NdArray::new(vec![vec![vec![50, 50]], vec![vec![50]], vec![vec![50]]]);
}

#[test]
fn println() {
    println!("{:?}", NdArray::new([[[10, 20], [30, 40]]]));
    println!("{:?}", NdArray::new([vec![vec![5, 10], vec![500, 100]]]));
}

#[test]
fn index() {
    let a = NdArray::new([10, 20, 30, 40]);
    assert_eq!(a[0], 10);
    assert_eq!(a[3], 40);

    let a = NdArray::new([[10, 20], [30, 40]]);
    assert_eq!(a[[0, 1]], 20);
    assert_eq!(a[[1, 1]], 40);
}

#[test]
fn slice_along_1d() {
    let a = NdArray::new([10, 20, 30, 40]);

    let slice = a.slice_along(Axis(0), 1);
    assert_eq!(slice.len(), 0);
    assert_eq!(slice.shape(), &[]);
    assert_eq!(slice.ndims(), 0);

    let slice = a.slice_along(Axis(0), ..);
    assert_eq!(slice.len(), 4);
    assert_eq!(slice[0], 10);
    assert_eq!(slice[3], 40);
    assert_eq!(slice.shape(), &[4]);
    assert_eq!(slice.ndims(), 1);

    let slice = a.slice_along(Axis(0), 2..);
    assert_eq!(slice.len(), 2);
    assert_eq!(slice[0], 30);
    assert_eq!(slice[1], 40);
    assert_eq!(slice.shape(), &[2]);
    assert_eq!(slice.ndims(), 1);

    let slice = a.slice_along(Axis(0), ..3);
    assert_eq!(slice.len(), 3);
    assert_eq!(slice[0], 10);
    assert_eq!(slice[2], 30);
    assert_eq!(slice.shape(), &[3]);
    assert_eq!(slice.ndims(), 1);

    let slice = a.slice_along(Axis(0), ..=3);
    assert_eq!(slice.len(), 4);
    assert_eq!(slice[0], 10);
    assert_eq!(slice[3], 40);
    assert_eq!(slice.shape(), &[4]);
    assert_eq!(slice.ndims(), 1);

    let slice = a.slice_along(Axis(0), 1..3);
    assert_eq!(slice.len(), 2);
    assert_eq!(slice[0], 20);
    assert_eq!(slice[1], 30);
    assert_eq!(slice.shape(), &[2]);
    assert_eq!(slice.ndims(), 1);

    let slice = a.slice_along(Axis(0), 1..=3);
    assert_eq!(slice.len(), 3);
    assert_eq!(slice[0], 20);
    assert_eq!(slice[2], 40);
    assert_eq!(slice.shape(), &[3]);
    assert_eq!(slice.ndims(), 1);
}

#[test]
fn slice_along_nd() {
    let a = NdArray::new([[10], [20], [30], [40]]);

    let slice = a.slice_along(Axis(0), 1);
    assert_eq!(slice.len(), 1);
    assert_eq!(slice.shape(), &[1]);
    assert_eq!(slice.ndims(), 1);
    assert_eq!(slice[0], 20);

    let slice = a.slice_along(Axis(1), 0);
    assert_eq!(slice.len(), 4);
    assert_eq!(slice.shape(), &[4]);
    assert_eq!(slice.ndims(), 1);
    assert_eq!(slice[0], 10);

    let a = NdArray::new([
        [[10, 20, 30], [40, 50, 60]],
        [[70, 80, 90], [100, 110, 120]],
    ]);

    let slice = a.slice_along(Axis(2), 2);
    assert_eq!(slice.len(), 2);
    assert_eq!(slice.shape(), &[2, 2]);
    assert_eq!(slice.ndims(), 2);
    assert_eq!(slice[[0, 0]], 30);
    assert_eq!(slice[[1, 0]], 90);

    let slice = a.slice_along(Axis(1), 1);
    assert_eq!(slice.len(), 2);
    assert_eq!(slice.shape(), &[2, 3]);
    assert_eq!(slice.ndims(), 2);
    assert_eq!(slice[[0, 0]], 40);
    assert_eq!(slice[[1, 2]], 120);

    let slice = a.slice_along(Axis(2), 1..);

    assert_eq!(slice.len(), 2);
    assert_eq!(slice.shape(), &[2, 2, 2]);
    assert_eq!(slice.ndims(), 3);

    assert_eq!(slice[[0, 0, 0]], 20);
    assert_eq!(slice[[0, 0, 1]], 30);
    assert_eq!(slice[[0, 1, 0]], 50);
    assert_eq!(slice[[1, 0, 0]], 80);
    assert_eq!(slice[[1, 1, 1]], 120);

    let slice = a.slice_along(Axis(1), 1..);

    assert_eq!(slice.len(), 2);
    assert_eq!(slice.shape(), &[2, 1, 3]);
    assert_eq!(slice.ndims(), 3);

    assert_eq!(slice[[0, 0, 0]], 40);
    assert_eq!(slice[[0, 0, 2]], 60);
    assert_eq!(slice[[1, 0, 0]], 100);
    assert_eq!(slice[[1, 0, 2]], 120);
}

#[test]
fn slice_homogenous() {
    let a = NdArray::new([
        [[1, 2, 3], [4, 5, 6]],
        [[7, 8, 9], [10, 11, 12]],
    ]);

    let slice = a.slice([1, 1]);

    assert_eq!(slice.len(), 3);
    assert_eq!(slice.shape(), &[3]);
    assert_eq!(slice.ndims(), 1);

    assert_eq!(slice[0], 10);
    assert_eq!(slice[1], 11);
    assert_eq!(slice[2], 12);

    let slice = a.slice([1..=1, 1..=1]);

    assert_eq!(slice.len(), 1);
    assert_eq!(slice.shape(), &[1, 1, 3]);
    assert_eq!(slice.ndims(), 3);

    assert_eq!(slice[[0, 0, 0]], 10);
    assert_eq!(slice[[0, 0, 1]], 11);
    assert_eq!(slice[[0, 0, 2]], 12);

    let slice = a.slice([0..=0, 0..=1, 0..=1]);

    assert_eq!(slice.len(), 1);
    assert_eq!(slice.shape(), &[1, 2, 2]);
    assert_eq!(slice.ndims(), 3);

    assert_eq!(slice[[0, 0, 0]], 1);
    assert_eq!(slice[[0, 0, 1]], 2);
    assert_eq!(slice[[0, 1, 0]], 4);
    assert_eq!(slice[[0, 1, 1]], 5);
}

#[test]
fn slice_heterogeneous() {
    let a = NdArray::new([
        [[1, 2, 3], [4, 5, 6]],
        [[7, 8, 9], [10, 11, 12]],
        [[13, 14, 15], [16, 17, 18]],
    ]);

    let slice = a.slice(s![0, .., 0..=1]);

    assert_eq!(slice.len(), 2);
    assert_eq!(slice.shape(), &[2, 2]);
    assert_eq!(slice.ndims(), 2);
}

#[test]
fn clone() {
    let arr;
    {
        let temp = NdArray::new([[[10, 20, 30]], [[40, 50, 60]]]);
        arr = temp.clone();
    }

    assert_eq!(arr.len(), 2);
    assert_eq!(arr.shape(), &[2, 1, 3]);
    assert_eq!(arr.stride(), &[3, 3, 1]);
    assert_eq!(arr.ndims(), 3);

    assert_eq!(arr[[0, 0, 0]], 10);
    assert_eq!(arr[[1, 0, 2]], 60);

    let arr2 = arr.slice(s![1, 0, ..]).clone();
    drop(arr);

    assert_eq!(arr2.len(), 3);
    assert_eq!(arr2.shape(), &[3]);
    assert_eq!(arr2.stride(), &[1]);
    assert_eq!(arr2.ndims(), 1);

    assert_eq!(arr2[0], 40);
    assert_eq!(arr2[1], 50);
    assert_eq!(arr2[2], 60);
}

#[test]
fn clone_contiguous() {
    let a = NdArray::new([
        [[10, 11, 12], [13, 14, 15]],
        [[16, 17, 18], [19, 20, 21]],
        [[22, 23, 24], [25, 26, 27]],
    ]);

    let view = a.slice([..]);
    let _ = view.clone();

    assert_eq!(view[[0, 0, 0]], 10);
    assert_eq!(view[[2, 1, 2]], 27);

    let a = NdArray::new(vec![5; 10]);
    let view = a.slice([..]);
    let _ = view.clone();

    assert_eq!(view[0], 5);
    assert_eq!(view[9], 5);
}

#[test]
fn flat_iter() {
    let a = NdArray::new([
        [[10, 11, 12], [13, 14, 15]],
        [[16, 17, 18], [19, 20, 21]],
        [[22, 23, 24], [25, 26, 27]],
    ]);

    let slice: Vec<_> = a.flatiter().collect();
    assert_eq!(slice, [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27]);

    let b = a.slice(s![.., 0]);
    let slice: Vec<_> = b.flatiter().collect();
    assert_eq!(slice, [10, 11, 12, 16, 17, 18, 22, 23, 24]);

    let b = a.slice(s![1]);
    let slice: Vec<_> = b.flatiter().collect();
    assert_eq!(slice, [16, 17, 18, 19, 20, 21]);

    let b = a.slice(s![..2, 1, 1..]);
    let slice: Vec<_> = b.flatiter().collect();
    assert_eq!(slice, [14, 15, 20, 21]);
}

#[test]
fn iterate() {
    let a = NdArray::new([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]);

    assert_eq!(a.iter().count(), 2);
    assert_eq!(a.iter().next().unwrap(), NdArray::new([[1, 2, 3], [4, 5, 6]]));
    assert_eq!(a.iter().last().unwrap(), NdArray::new([[7, 8, 9], [10, 11, 12]]));

    assert_eq!(a.iter_along(1).count(), 2);
    assert_eq!(a.iter_along(Axis(1)).next().unwrap(), NdArray::new([[1, 2, 3], [7, 8, 9]]));
    assert_eq!(a.iter_along(Axis(1)).last().unwrap(), NdArray::new([[4, 5, 6], [10, 11, 12]]));

    assert_eq!(a.iter_along(2).count(), 3);
    assert_eq!(a.iter_along(Axis(2)).next().unwrap(), NdArray::new([[1, 4], [7, 10]]));
    assert_eq!(a.iter_along(Axis(2)).last().unwrap(), NdArray::new([[3, 6], [9, 12]]));

    assert_eq!(a.nditer([0, 1]).count(), 4);
    assert_eq!(a.nditer([0, 1]).next().unwrap(), NdArray::new([1, 2, 3]));
    assert_eq!(a.nditer(vec![0, 1]).last().unwrap(), NdArray::new([10, 11, 12]));

    assert_eq!(a.nditer([0, 2]).count(), 6);
    assert_eq!(a.nditer([0, 2]).next().unwrap(), NdArray::new([1, 4]));
    assert_eq!(a.nditer(vec![0, 2]).last().unwrap(), NdArray::new([9, 12]));

    assert_eq!(a.nditer([1, 2]).count(), 6);
    assert_eq!(a.nditer([1, 2]).next().unwrap(), NdArray::new([1, 7]));
    assert_eq!(a.nditer(vec![1, 2]).last().unwrap(), NdArray::new([6, 12]));

    assert_eq!(a.nditer([0, 1, 2]).count(), 12);
    assert_eq!(a.nditer([0, 1, 2]).next().unwrap(), NdArray::scalar(1));
    assert_eq!(a.nditer(vec![0, 1, 2]).last().unwrap(), NdArray::scalar(12));
}
