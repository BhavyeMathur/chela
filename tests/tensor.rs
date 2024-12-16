use chela::*;

#[test]
fn from_vector() {
    let arr = Tensor::from(vec![0, 50, 100]);
    assert_eq!(arr.len(), &3);
    assert_eq!(arr.shape(), &[3]);
    assert_eq!(arr.stride(), &[1]);
    assert_eq!(arr.ndims(), 1);
    assert_eq!(arr.size(), 3);

    let arr = Tensor::from(vec![vec![50], vec![50], vec![50]]);
    assert_eq!(arr.len(), &3);
    assert_eq!(arr.shape(), &[3, 1]);
    assert_eq!(arr.stride(), &[1, 1]);
    assert_eq!(arr.ndims(), 2);
    assert_eq!(arr.size(), 3);

    let arr = Tensor::from(vec![vec![vec![50]], vec![vec![50]]]);
    assert_eq!(arr.len(), &2);
    assert_eq!(arr.shape(), &[2, 1, 1]);
    assert_eq!(arr.stride(), &[1, 1, 1]);
    assert_eq!(arr.ndims(), 3);
    assert_eq!(arr.size(), 2);

    let arr = Tensor::from(vec![vec![vec![50, 50, 50]], vec![vec![50, 50, 50]]]);
    assert_eq!(arr.len(), &2);
    assert_eq!(arr.shape(), &[2, 1, 3]);
    assert_eq!(arr.stride(), &[3, 3, 1]);
    assert_eq!(arr.ndims(), 3);
    assert_eq!(arr.size(), 6);
}

#[test]
fn from_array() {
    let arr = Tensor::from([500, 50, 100]);
    assert_eq!(arr.len(), &3);
    assert_eq!(arr.shape(), &[3]);
    assert_eq!(arr.stride(), &[1]);
    assert_eq!(arr.ndims(), 1);

    let arr = Tensor::from([[500], [50], [100]]);
    assert_eq!(arr.len(), &3);
    assert_eq!(arr.shape(), &[3, 1]);
    assert_eq!(arr.stride(), &[1, 1]);
    assert_eq!(arr.ndims(), 2);

    let arr = Tensor::from([[[500], [50], [30]], [[50], [0], [0]]]);
    assert_eq!(arr.len(), &2);
    assert_eq!(arr.shape(), &[2, 3, 1]);
    assert_eq!(arr.stride(), &[3, 1, 1]);
    assert_eq!(arr.ndims(), 3);

    let arr = Tensor::from([[[50, 50, 50]], [[50, 50, 50]]]);
    assert_eq!(arr.len(), &2);
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
    assert_eq!(slice.len(), &0);
    assert_eq!(slice.shape(), &[]);
    assert_eq!(slice.ndims(), 0);

    let slice = a.slice_along(Axis(0), ..);
    assert_eq!(slice.len(), &4);
    assert_eq!(slice[0], 10);
    assert_eq!(slice[3], 40);
    assert_eq!(slice.shape(), &[4]);
    assert_eq!(slice.ndims(), 1);

    let slice = a.slice_along(Axis(0), 2..);
    assert_eq!(slice.len(), &2);
    assert_eq!(slice[0], 30);
    assert_eq!(slice[1], 40);
    assert_eq!(slice.shape(), &[2]);
    assert_eq!(slice.ndims(), 1);

    let slice = a.slice_along(Axis(0), ..3);
    assert_eq!(slice.len(), &3);
    assert_eq!(slice[0], 10);
    assert_eq!(slice[2], 30);
    assert_eq!(slice.shape(), &[3]);
    assert_eq!(slice.ndims(), 1);

    let slice = a.slice_along(Axis(0), ..=3);
    assert_eq!(slice.len(), &4);
    assert_eq!(slice[0], 10);
    assert_eq!(slice[3], 40);
    assert_eq!(slice.shape(), &[4]);
    assert_eq!(slice.ndims(), 1);

    let slice = a.slice_along(Axis(0), 1..3);
    assert_eq!(slice.len(), &2);
    assert_eq!(slice[0], 20);
    assert_eq!(slice[1], 30);
    assert_eq!(slice.shape(), &[2]);
    assert_eq!(slice.ndims(), 1);

    let slice = a.slice_along(Axis(0), 1..=3);
    assert_eq!(slice.len(), &3);
    assert_eq!(slice[0], 20);
    assert_eq!(slice[2], 40);
    assert_eq!(slice.shape(), &[3]);
    assert_eq!(slice.ndims(), 1);
}

#[test]
fn slice_along_nd() {
    let a = Tensor::from([[10], [20], [30], [40]]);

    let slice = a.slice_along(Axis(0), 1);
    assert_eq!(slice.len(), &1);
    assert_eq!(slice.shape(), &[1]);
    assert_eq!(slice.ndims(), 1);
    assert_eq!(slice[0], 20);

    let slice = a.slice_along(Axis(1), 0);
    assert_eq!(slice.len(), &4);
    assert_eq!(slice.shape(), &[4]);
    assert_eq!(slice.ndims(), 1);
    assert_eq!(slice[0], 10);

    let a = Tensor::from([
        [[10, 20, 30], [40, 50, 60]],
        [[70, 80, 90], [100, 110, 120]],
    ]);

    let slice = a.slice_along(Axis(2), 2);
    assert_eq!(slice.len(), &2);
    assert_eq!(slice.shape(), &[2, 2]);
    assert_eq!(slice.ndims(), 2);
    assert_eq!(slice[[0, 0]], 30);
    assert_eq!(slice[[1, 0]], 90);

    let slice = a.slice_along(Axis(1), 1);
    assert_eq!(slice.len(), &2);
    assert_eq!(slice.shape(), &[2, 3]);
    assert_eq!(slice.ndims(), 2);
    assert_eq!(slice[[0, 0]], 40);
    assert_eq!(slice[[1, 2]], 120);

    let slice = a.slice_along(Axis(2), 1..);

    assert_eq!(slice.len(), &2);
    assert_eq!(slice.shape(), &[2, 2, 2]);
    assert_eq!(slice.ndims(), 3);

    assert_eq!(slice[[0, 0, 0]], 20);
    assert_eq!(slice[[0, 0, 1]], 30);
    assert_eq!(slice[[0, 1, 0]], 50);
    assert_eq!(slice[[1, 0, 0]], 80);
    assert_eq!(slice[[1, 1, 1]], 120);

    let slice = a.slice_along(Axis(1), 1..);

    assert_eq!(slice.len(), &2);
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

    assert_eq!(slice.len(), &3);
    assert_eq!(slice.shape(), &[3]);
    assert_eq!(slice.ndims(), 1);

    assert_eq!(slice[0], 10);
    assert_eq!(slice[1], 11);
    assert_eq!(slice[2], 12);

    let slice = a.slice([1..=1, 1..=1]);

    assert_eq!(slice.len(), &1);
    assert_eq!(slice.shape(), &[1, 1, 3]);
    assert_eq!(slice.ndims(), 3);

    assert_eq!(slice[[0, 0, 0]], 10);
    assert_eq!(slice[[0, 0, 1]], 11);
    assert_eq!(slice[[0, 0, 2]], 12);

    let slice = a.slice([0..=0, 0..=1, 0..=1]);

    assert_eq!(slice.len(), &1);
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

    assert_eq!(slice.len(), &2);
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

    assert_eq!(arr.len(), &2);
    assert_eq!(arr.shape(), &[2, 1, 3]);
    assert_eq!(arr.stride(), &[3, 3, 1]);
    assert_eq!(arr.ndims(), 3);

    assert_eq!(arr[[0, 0, 0]], 10);
    assert_eq!(arr[[1, 0, 2]], 60);

    let arr2 = arr.slice(s![1, 0, ..]).clone();
    drop(arr);

    assert_eq!(arr2.len(), &3);
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
    view.clone();

    assert_eq!(view[[0, 0, 0]], 10);
    assert_eq!(view[[2, 1, 2]], 27);

    let a = Tensor::from(vec![5; 10]);
    let view = a.slice([..]);
    view.clone();

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
    assert_eq!(b.len(), &18);
    assert_eq!(b.ndims(), 1);

    assert_eq!(b[0], 10);
    assert_eq!(b[5], 15);
    assert_eq!(b[17], 27);

    let b = a.slice(s![.., 0]).flatten();
    assert_eq!(b.shape(), &[9]);
    assert_eq!(b.stride(), &[1]);
    assert_eq!(b.len(), &9);
    assert_eq!(b.ndims(), 1);

    assert_eq!(b[0], 10);
    assert_eq!(b[5], 18);
    assert_eq!(b[8], 24);

    let b = a.slice(s![..2, 1, 1..]).flatten();
    assert_eq!(b.shape(), &[4]);
    assert_eq!(b.stride(), &[1]);
    assert_eq!(b.len(), &4);
    assert_eq!(b.ndims(), 1);

    assert_eq!(b[0], 14);
    assert_eq!(b[3], 21);
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
fn full_i32(){
    let a = Tensor::full(3, vec![2, 3]);
    assert_eq!(a.shape(), &[2, 3]);
    assert_eq!(a.stride(), &[3, 1]);
    let b = a.flatten();
    let b_len = b.len().clone();
    for i in 0..b_len {
        assert_eq!(b[i], 3i32);
    }
}

#[test]
fn full_f64(){
    let a = Tensor::full(3.2,vec![4, 6, 2]);
    assert_eq!(a.shape(), &[4, 6, 2]);
    let b = a.flatten();
    let b_len = *b.len();
    for i in 0..b_len {
        assert_eq!(b[i], 3.2f64);
    }
}

#[test]
fn full_bool(){
    let a: Tensor<bool> = Tensor::full(true,vec![3, 5, 3]);
    assert_eq!(a.shape(), &[3, 5, 3]);
    assert_eq!(a.stride(), &[15, 3, 1]);
    let b = a.flatten();
    let b_len = *b.len();
    for i in 0..b_len {
        assert_eq!(b[i], true);
    }
}

#[test]
fn ones_u8(){
    let a: Tensor<u8> = Tensor::ones(vec![3, 5, 3]);
    assert_eq!(a.shape(), &[3, 5, 3]);
    assert_eq!(a.stride(), &[15, 3, 1]);
    let b = a.flatten();
    let b_len = *b.len();
    for i in 0..b_len {
        assert_eq!(b[i], 1u8);
    }
}

#[test]
fn ones_i32(){
    let a: Tensor<i32> = Tensor::ones(vec![3, 5, 3]);
    assert_eq!(a.shape(), &[3, 5, 3]);
    assert_eq!(a.stride(), &[15, 3, 1]);
    let b = a.flatten();
    let b_len = *b.len();
    for i in 0..b_len {
        assert_eq!(b[i], 1i32);
    }
}

#[test]
fn ones_1d(){
    let a: Tensor<u8> = Tensor::ones(vec![4]);
    assert_eq!(a.shape(), &[4]);
    let a_len = *a.len();
    for i in 0..a_len {
        assert_eq!(a[i], 1u8);
    }
}

#[test]
fn ones_f64(){
    let a: Tensor<f64> = Tensor::ones(vec![4]);
    assert_eq!(a.shape(), &[4]);
    let a_len = *a.len();
    for i in 0..a_len {
        assert_eq!(a[i], 1f64);
    }
}

#[test]
fn ones_bool(){
    let a: Tensor<bool> = Tensor::ones(vec![3, 5, 3]);
    assert_eq!(a.shape(), &[3, 5, 3]);
    assert_eq!(a.stride(), &[15, 3, 1]);
    let b = a.flatten();
    let b_len = *b.len();
    for i in 0..b_len {
        assert_eq!(b[i], true);
    }
}

#[test]
fn zeroes_u8(){
    let a: Tensor<u8> = Tensor::zeros(vec![3, 5, 3]);
    assert_eq!(a.shape(), &[3, 5, 3]);
    assert_eq!(a.stride(), &[15, 3, 1]);
    let b = a.flatten();
    let b_len = *b.len();
    for i in 0..b_len {
        assert_eq!(b[i], 0u8);
    }
}

#[test]
fn zeroes_i32(){
    let a: Tensor<i32> = Tensor::zeros(vec![3, 5, 3]);
    assert_eq!(a.shape(), &[3, 5, 3]);
    assert_eq!(a.stride(), &[15, 3, 1]);
    let b = a.flatten();
    let b_len = *b.len();
    for i in 0..b_len {
        assert_eq!(b[i], 0i32);
    }
}

#[test]
fn zeroes_1d(){
    let a: Tensor<u8> = Tensor::zeros(vec![4]);
    assert_eq!(a.shape(), &[4]);
    let a_len = *a.len();
    for i in 0..a_len {
        assert_eq!(a[i], 0u8);
    }
}

#[test]
fn zeroes_f64(){
    let a: Tensor<f64> = Tensor::zeros(vec![4]);
    assert_eq!(a.shape(), &[4]);
    let a_len = *a.len();
    for i in 0..a_len {
        assert_eq!(a[i], 0f64);
    }
}

#[test]
fn zeroes_bool(){
    let a: Tensor<bool> = Tensor::zeros(vec![3, 5, 3]);
    assert_eq!(a.shape(), &[3, 5, 3]);
    assert_eq!(a.stride(), &[15, 3, 1]);
    let b = a.flatten();
    let b_len = *b.len();
    for i in 0..b_len {
        assert_eq!(b[i], false);
    }
}