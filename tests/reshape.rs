use redstone::*;

#[test]
fn flatten() {
    let a = NdArray::new([
        [[10, 11, 12], [13, 14, 15]],
        [[16, 17, 18], [19, 20, 21]],
        [[22, 23, 24], [25, 26, 27]],
    ]);

    let b = a.flatten();
    assert_eq!(b.shape(), &[18]);
    assert_eq!(b.stride(), &[1]);
    assert_eq!(b.len(), 18);
    assert_eq!(b.ndims(), 1);

    let correct = NdArray::new([10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
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

    let correct = NdArray::new([14, 15, 20, 21]);
    assert_eq!(b, correct);
}

#[test]
fn squeeze_first_dimension() {
    let a = NdArray::new([
        [[[1, 2, 3], [4, 5, 6]]],
    ]);
    let b = a.squeeze();
    assert_eq!(b.shape(), &[2, 3]);
    assert_eq!(b.stride(), &[3, 1]);
}

#[test]
fn squeeze_multiple_dimensions() {
    let a = NdArray::new([
        [[[[1, 2, 3]], [[4, 5, 6]]]],
    ]);
    let b = a.squeeze();
    assert_eq!(b.shape(), &[2, 3]);
    assert_eq!(b.stride(), &[3, 1]);
}

#[test]
fn squeeze_one_dimension() {
    let a = NdArray::new([0]);
    let b = a.squeeze();
    assert_eq!(b.shape(), &[]);
    assert_eq!(b.stride(), &[]);
}

#[test]
fn squeeze_no_extra_dimensions() {
    let a = NdArray::new([[1, 2, 3], [4, 5, 6]]);
    let b = a.squeeze();
    assert_eq!(b.shape(), &[2, 3]);
    assert_eq!(b.stride(), &[3, 1]);
}

#[test]
fn unsqueeze_single_element() {
    let a = NdArray::new([0]);
    let b = a.unsqueeze(Axis(0));
    assert_eq!(b.shape(), &[1, 1]);
    assert_eq!(b.stride(), &[1, 1]);
}

#[test]
fn unsqueeze_random_dimension_first_axis() {
    let a = NdArray::new([[1, 2, 3], [4, 5, 6]]);
    let b = a.unsqueeze(Axis(0));
    assert_eq!(b.shape(), &[1, 2, 3]);
    assert_eq!(b.stride(), &[6, 3, 1]);
}

#[test]
fn unsqueeze_random_dimension_axis_1() {
    let a = NdArray::new([[1, 2, 3], [4, 5, 6]]);
    let b = a.unsqueeze(Axis(1));
    assert_eq!(b.shape(), &[2, 1, 3]);
    assert_eq!(b.stride(), &[3, 3, 1]);
}

#[test]
fn unsqueeze_random_dimension_last_axis() {
    let a = NdArray::new([[1, 2, 3], [4, 5, 6]]);
    let b = a.unsqueeze(Axis(2));
    assert_eq!(b.shape(), &[2, 3, 1]);
    assert_eq!(b.stride(), &[3, 1, 1]);
}

#[test]
fn test_view() {
    let tensor = NdArray::randint([3, 2, 4], 0, 255);

    let view = (&tensor).view();
    assert_eq!(tensor, view);
    assert_eq!(tensor, view.view());
}

#[test]
fn test_view_slice() {
    let tensor = NdArray::randint([3, 2, 4], 0, 255);

    // uniformly strided slice
    let slice = tensor.slice(s![0]);
    assert_eq!(slice, (&slice).view());

    // non-uniformly strided slice
    let slice = tensor.slice(s![0..2, .., 0]);
    assert_eq!(slice, (&slice).view());
}

#[test]
fn test_reshape() {
    let tensor = NdArray::new([
        [[1, 2], [3, 4], [5, 6]],
        [[1, 2], [3, 4], [5, 6]],
        [[1, 2], [3, 4], [5, 6]],
        [[1, 2], [3, 4], [5, 6]]]); // [4, 3, 2]

    let flat = (&tensor).reshape([4 * 3 * 2]);
    assert_eq!(flat, NdArray::new([1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6]));

    let view1 = (&tensor).reshape([4, 3 * 2]);
    assert_eq!(view1, NdArray::new([[1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6]]));

    let view2 = (&tensor).reshape([3, 8]);
    assert_eq!(view2, NdArray::new([[1, 2, 3, 4, 5, 6, 1, 2], [3, 4, 5, 6, 1, 2, 3, 4], [5, 6, 1, 2, 3, 4, 5, 6]]));
}

#[test]
fn test_reshape_slice() {
    let tensor = NdArray::new([
        [[1, 2], [30, 40], [5, 6]],
        [[1, 2], [31, 41], [5, 6]],
        [[1, 2], [32, 42], [5, 6]],
        [[1, 2], [33, 43], [5, 6]]]); // [4, 3, 2]

    // uniformly strided slice
    let slice = tensor.slice(s![.., .., 0]);  // [4, 3] (stride: [6, 2])

    let view = (&slice).reshape([4, 3]); // (stride: [6, 2])
    assert_eq!(view, NdArray::new([[1, 30, 5], [1, 31, 5], [1, 32, 5], [1, 33, 5]]));

    let view = (&slice).reshape([3, 4]); // (stride: [8, 2])
    assert_eq!(view, NdArray::new([[1, 30, 5, 1], [31, 5, 1, 32], [5, 1, 33, 5]]));

    let view = (&slice).reshape([2, 6]); // (stride: [12, 2])
    assert_eq!(view, NdArray::new([[1, 30, 5, 1, 31, 5], [1, 32, 5, 1, 33, 5]]));

    // uniformly strided slice TODO this is not supported right now but can be
    // let slice = tensor.slice(s![.., 1]);  // [4, 2] (stride: [6, 1])
    // let view = (&slice).reshape([2, 2, 2]);  // (stride: [12, 6, 1])
    // assert_eq!(view, NdArray::new([[[30, 40], [31, 41]], [[32, 42], [33, 43]]]));
}

#[test]
fn test_reshape_identity() {
    let tensor = NdArray::new([
        [[1, 2], [30, 40], [5, 6]],
        [[1, 2], [31, 41], [5, 6]],
        [[1, 2], [32, 42], [5, 6]],
        [[1, 2], [33, 43], [5, 6]]]); // [4, 3, 2]
    
    // otherwise not reshapable
    let slice = tensor.slice(s![.., 0..=1]);  // [4, 2, 2]
    let view = (&slice).reshape([4, 2, 2]);
    
    assert_eq!(view, slice);
}

#[test]
#[should_panic]
fn test_reshape_invalid_stride() {
    let tensor = NdArray::new([
        [[1, 2], [30, 40], [5, 6]],
        [[1, 2], [31, 41], [5, 6]],
        [[1, 2], [32, 42], [5, 6]],
        [[1, 2], [33, 43], [5, 6]]]); // [4, 3, 2]

    let slice = tensor.slice(s![.., 1]);  // [4, 2]
    slice.reshape([4 * 2]);
}

#[test]
#[should_panic]
fn test_reshape_invalid_shape() {
    let tensor = NdArray::new([
        [[1, 2], [30, 40], [5, 6]],
        [[1, 2], [31, 41], [5, 6]],
        [[1, 2], [32, 42], [5, 6]],
        [[1, 2], [33, 43], [5, 6]]]); // [4, 3, 2]

    tensor.reshape([4 * 2]);
}
