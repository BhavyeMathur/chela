use chela::*;
use paste::paste;


#[test]
#[should_panic]
fn test_diagonal_invalid_axis1() {
    let a = Tensor::arange(0, 12);
    let a = a.reshape([3, 4]);
    a.diagonal_along(3, 0);
}

#[test]
#[should_panic]
fn test_diagonal_invalid_axis2() {
    let a = Tensor::arange(0, 12);
    let a = a.reshape([3, 4]);
    a.diagonal_along(0, 3);
}

#[test]
#[should_panic]
fn test_diagonal_invalid_axes() {
    let a = Tensor::arange(0, 12);
    let a = a.reshape([3, 4]);
    a.diagonal_along(0, 0);
}

test_for_all_numeric_dtypes!(
    test_diagonal, {
        let a = Tensor::arange(0, 12).astype::<T>();
        let a = a.reshape([3, 4]);

        let expected = Tensor::from([0, 5, 10]).astype::<T>();
        assert_eq!(a.diagonal(), expected);
        assert!(a.diagonal().is_view());

        assert_eq!(a.diagonal_along(0, 1), expected);
        assert_eq!(a.diagonal_along(-2, 1), expected);
        assert_eq!(a.offset_diagonal(0), expected);

        let expected = Tensor::from([1, 6, 11]).astype::<T>();
        assert_eq!(a.offset_diagonal(1), expected);

        let expected = Tensor::from([4, 9]).astype::<T>();
        assert_eq!(a.offset_diagonal(-1), expected);
    }
);

test_for_all_numeric_dtypes!(
    test_diagonal_3d, {
        let a = Tensor::arange(0, 8).astype::<T>();
        let a = a.reshape([2, 2, 2]);

        let expected = Tensor::from([[0, 6], [1, 7]]).astype::<T>();
        assert_eq!(a.diagonal(), expected);

        let expected = Tensor::from([[2], [3]]).astype::<T>();
        assert_eq!(a.offset_diagonal(1), expected);

        let expected = Tensor::from([[4], [5]]).astype::<T>();
        assert_eq!(a.offset_diagonal(-1), expected);

        let expected = Tensor::from([[0, 3], [4, 7]]).astype::<T>();
        assert_eq!(a.diagonal_along(1, 2), expected);
        assert_eq!(a.diagonal_along(2, 1), expected);

        let expected = Tensor::from([[1], [3]]).astype::<T>();
        assert_eq!(a.offset_diagonal_along(1, Axis(0), Axis(2)), expected);
    }
);

test_for_all_numeric_dtypes!(
    test_trace, {
        let a = Tensor::arange(0, 12).astype::<T>();
        let a = a.reshape([3, 4]);

        let expected = Tensor::scalar(15).astype::<T>();
        assert_eq!(a.trace(), expected);

        assert_eq!(a.trace_along(0, 1), expected);
        assert_eq!(a.trace_along(-2, 1), expected);
        assert_eq!(a.offset_trace(0), expected);

        let expected = Tensor::scalar(18).astype::<T>();
        assert_eq!(a.offset_trace(1), expected);

        let expected = Tensor::scalar(13).astype::<T>();
        assert_eq!(a.offset_trace(-1), expected);
    }
);


test_for_all_numeric_dtypes!(
    test_trace_3d, {
        let a = Tensor::arange(0, 8).astype::<T>();
        let a = a.reshape([2, 2, 2]);

        let expected = Tensor::from([6, 8]).astype::<T>();
        assert_eq!(a.trace(), expected);

        let expected = Tensor::from([2, 3]).astype::<T>();
        assert_eq!(a.offset_trace(1), expected);

        let expected = Tensor::from([4, 5]).astype::<T>();
        assert_eq!(a.offset_trace(-1), expected);

        let expected = Tensor::from([3, 11]).astype::<T>();
        assert_eq!(a.trace_along(1, 2), expected);
        assert_eq!(a.trace_along(2, 1), expected);

        let expected = Tensor::from([1, 3]).astype::<T>();
        assert_eq!(a.offset_trace_along(1, Axis(0), Axis(2)), expected);
    }
);
