use chela::*;

#[test]
fn from_vector() {
    let arr = Tensor::from(vec![0, 50, 100]);
    assert_eq!(arr.len(), 3);
    assert_eq!(arr.shape(), &[3]);
    assert_eq!(arr.stride(), &[1]);
    assert_eq!(arr.ndims(), 1);
    assert_eq!(arr.size(), 3);

    let arr = Tensor::from(vec![vec![50], vec![50], vec![50]]);
    assert_eq!(arr.len(), 3);
    assert_eq!(arr.shape(), &[3, 1]);
    assert_eq!(arr.stride(), &[1, 1]);
    assert_eq!(arr.ndims(), 2);
    assert_eq!(arr.size(), 3);

    let arr = Tensor::from(vec![vec![vec![50]], vec![vec![50]]]);
    assert_eq!(arr.len(), 2);
    assert_eq!(arr.shape(), &[2, 1, 1]);
    assert_eq!(arr.stride(), &[1, 1, 1]);
    assert_eq!(arr.ndims(), 3);
    assert_eq!(arr.size(), 2);

    let arr = Tensor::from(vec![vec![vec![50, 50, 50]], vec![vec![50, 50, 50]]]);
    assert_eq!(arr.len(), 2);
    assert_eq!(arr.shape(), &[2, 1, 3]);
    assert_eq!(arr.stride(), &[3, 3, 1]);
    assert_eq!(arr.ndims(), 3);
    assert_eq!(arr.size(), 6);
}

#[test]
fn from_array() {
    let arr = Tensor::from([500, 50, 100]);
    assert_eq!(arr.len(), 3);
    assert_eq!(arr.shape(), &[3]);
    assert_eq!(arr.stride(), &[1]);
    assert_eq!(arr.ndims(), 1);

    let arr = Tensor::from([[500], [50], [100]]);
    assert_eq!(arr.len(), 3);
    assert_eq!(arr.shape(), &[3, 1]);
    assert_eq!(arr.stride(), &[1, 1]);
    assert_eq!(arr.ndims(), 2);

    let arr = Tensor::from([[[500], [50], [30]], [[50], [0], [0]]]);
    assert_eq!(arr.len(), 2);
    assert_eq!(arr.shape(), &[2, 3, 1]);
    assert_eq!(arr.stride(), &[3, 1, 1]);
    assert_eq!(arr.ndims(), 3);

    let arr = Tensor::from([[[50, 50, 50]], [[50, 50, 50]]]);
    assert_eq!(arr.len(), 2);
    assert_eq!(arr.shape(), &[2, 1, 3]);
    assert_eq!(arr.stride(), &[3, 3, 1]);
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
    let a = Tensor::from([[10], [20], [30], [40]]);

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

    let a = Tensor::from([
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
    let a = Tensor::from([
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
    let a = Tensor::from([
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
        let temp = Tensor::from([[[10, 20, 30]], [[40, 50, 60]]]);
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
    let a = Tensor::from([
        [[10, 11, 12], [13, 14, 15]],
        [[16, 17, 18], [19, 20, 21]],
        [[22, 23, 24], [25, 26, 27]],
    ]);

    let view = a.slice([..]);
    let _ = view.clone();

    assert_eq!(view[[0, 0, 0]], 10);
    assert_eq!(view[[2, 1, 2]], 27);

    let a = Tensor::from(vec![5; 10]);
    let view = a.slice([..]);
    let _ = view.clone();

    assert_eq!(view[0], 5);
    assert_eq!(view[9], 5);
}

#[test]
fn flat_iter() {
    let a = Tensor::from([
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
fn flatten() {
    let a = Tensor::from([
        [[10, 11, 12], [13, 14, 15]],
        [[16, 17, 18], [19, 20, 21]],
        [[22, 23, 24], [25, 26, 27]],
    ]);

    let b = a.flatten();
    assert_eq!(b.shape(), &[18]);
    assert_eq!(b.stride(), &[1]);
    assert_eq!(b.len(), 18);
    assert_eq!(b.ndims(), 1);

    let correct = Tensor::from([10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
        20, 21, 22, 23, 24, 25, 26, 27]);
    assert_eq!(b, correct);

    let b = a.slice(s![.., 0]).flatten();
    assert_eq!(b.shape(), &[9]);
    assert_eq!(b.stride(), &[1]);
    assert_eq!(b.len(), 9);
    assert_eq!(b.ndims(), 1);

    assert_eq!(b[0], 10);
    assert_eq!(b[5], 18);
    assert_eq!(b[8], 24);

    let b = a.slice(s![..2, 1, 1..]).flatten();
    assert_eq!(b.shape(), &[4]);
    assert_eq!(b.stride(), &[1]);
    assert_eq!(b.len(), 4);
    assert_eq!(b.ndims(), 1);

    let correct = Tensor::from([14, 15, 20, 21]);
    assert_eq!(b, correct);
}

#[test]
fn squeeze_first_dimension() {
    let a = Tensor::from([
        [[[1, 2, 3], [4, 5, 6]]],
    ]);
    let b = a.squeeze();
    assert_eq!(b.shape(), &[2, 3]);
    assert_eq!(b.stride(), &[3, 1]);
}

#[test]
fn squeeze_multiple_dimensions() {
    let a = Tensor::from([
        [[[[1, 2, 3]], [[4, 5, 6]]]],
    ]);
    let b = a.squeeze();
    assert_eq!(b.shape(), &[2, 3]);
    assert_eq!(b.stride(), &[3, 1]);
}

#[test]
fn squeeze_one_dimension() {
    let a: Tensor<i32> = Tensor::from([0]);
    let b = a.squeeze();
    assert_eq!(b.shape(), &[]);
    assert_eq!(b.stride(), &[]);
}

#[test]
fn squeeze_no_extra_dimensions() {
    let a: Tensor<i32> = Tensor::from([[1, 2, 3], [4, 5, 6]]);
    let b = a.squeeze();
    assert_eq!(b.shape(), &[2, 3]);
    assert_eq!(b.stride(), &[3, 1]);
}

#[test]
fn unsqueeze_single_element() {
    let a: Tensor<i32> = Tensor::from([0]);
    let b = a.unsqueeze(Axis(0));
    assert_eq!(b.shape(), &[1, 1]);
    assert_eq!(b.stride(), &[1, 1]);
}

#[test]
fn unsqueeze_random_dimension_first_axis() {
    let a: Tensor<i32> = Tensor::from([[1, 2, 3], [4, 5, 6]]);
    let b = a.unsqueeze(Axis(0));
    assert_eq!(b.shape(), &[1, 2, 3]);
    assert_eq!(b.stride(), &[6, 3, 1]);
}

#[test]
fn unsqueeze_random_dimension_axis_1() {
    let a = Tensor::from([[1, 2, 3], [4, 5, 6]]);
    let b = a.unsqueeze(Axis(1));
    assert_eq!(b.shape(), &[2, 1, 3]);
    assert_eq!(b.stride(), &[3, 3, 1]);
}

#[test]
fn unsqueeze_random_dimension_last_axis() {
    let a = Tensor::from([[1, 2, 3], [4, 5, 6]]);
    let b = a.unsqueeze(Axis(2));
    assert_eq!(b.shape(), &[2, 3, 1]);
    assert_eq!(b.stride(), &[3, 1, 1]);
}

#[test]
fn iterate() {
    let a = Tensor::from([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]);

    assert_eq!(a.iter().count(), 2);
    assert_eq!(a.iter().next().unwrap(), Tensor::from([[1, 2, 3], [4, 5, 6]]));
    assert_eq!(a.iter().last().unwrap(), Tensor::from([[7, 8, 9], [10, 11, 12]]));

    assert_eq!(a.iter_along(1).count(), 2);
    assert_eq!(a.iter_along(Axis(1)).next().unwrap(), Tensor::from([[1, 2, 3], [7, 8, 9]]));
    assert_eq!(a.iter_along(Axis(1)).last().unwrap(), Tensor::from([[4, 5, 6], [10, 11, 12]]));

    assert_eq!(a.iter_along(2).count(), 3);
    assert_eq!(a.iter_along(Axis(2)).next().unwrap(), Tensor::from([[1, 4], [7, 10]]));
    assert_eq!(a.iter_along(Axis(2)).last().unwrap(), Tensor::from([[3, 6], [9, 12]]));

    assert_eq!(a.nditer([0, 1]).count(), 4);
    assert_eq!(a.nditer([0, 1]).next().unwrap(), Tensor::from([1, 2, 3]));
    assert_eq!(a.nditer(vec![0, 1]).last().unwrap(), Tensor::from([10, 11, 12]));

    assert_eq!(a.nditer([0, 2]).count(), 6);
    assert_eq!(a.nditer([0, 2]).next().unwrap(), Tensor::from([1, 4]));
    assert_eq!(a.nditer(vec![0, 2]).last().unwrap(), Tensor::from([9, 12]));

    assert_eq!(a.nditer([1, 2]).count(), 6);
    assert_eq!(a.nditer([1, 2]).next().unwrap(), Tensor::from([1, 7]));
    assert_eq!(a.nditer(vec![1, 2]).last().unwrap(), Tensor::from([6, 12]));

    assert_eq!(a.nditer([0, 1, 2]).count(), 12);
    assert_eq!(a.nditer([0, 1, 2]).next().unwrap(), Tensor::scalar(1));
    assert_eq!(a.nditer(vec![0, 1, 2]).last().unwrap(), Tensor::scalar(12));
}

#[test]
fn test_fill_f32() {
    let mut a: Tensor<f32> = Tensor::zeros([3, 5, 3]);

    assert!(a.flatiter().all(|x| x == 0.0));
    a.fill(25.0);
    assert!(a.flatiter().all(|x| x == 25.0));
}

#[test]
fn test_fill_f64() {
    let mut a: Tensor<f64> = Tensor::zeros([15]);

    assert!(a.flatiter().all(|x| x == 0.0));
    a.fill(20.0);
    assert!(a.flatiter().all(|x| x == 20.0));
}

#[test]
fn test_fill_slice() {
    let a: Tensor<i32> = Tensor::zeros([3, 5]);
    let correct = Tensor::from([[0, 5, 0, 0, 0], [0, 5, 0, 0, 0], [0, 5, 0, 0, 0]]);
    a.slice(s![.., 1]).fill(5);
    assert_eq!(a, correct);

    let a: Tensor<u32> = Tensor::zeros([3, 5]);
    let correct: Tensor<u32> = Tensor::from([[0, 5, 5, 5, 0], [0, 5, 5, 5, 0], [0, 5, 5, 5, 0]]);
    a.slice(s![.., 1..4]).fill(5);
    assert_eq!(a, correct);

    let a: Tensor<bool> = Tensor::zeros([3, 5]);
    let correct: Tensor<bool> = Tensor::from(
        [[false, false, false, false, false],
            [true, true, true, true, true],
            [false, false, false, false, false]]);
    a.slice(s![1, ..]).fill(true);
    assert_eq!(a, correct);
}

#[test]
fn test() {
    let a: Tensor<f32> = Tensor::rand(10);
    let out = a.sum_along(Axis(0));
    println!("{:?}", out);
}
