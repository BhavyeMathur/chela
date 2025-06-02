use chela::*;
use paste::paste;

test_for_all_numeric_dtypes!(
    test_full, {
        let a = NdArray::full(3 as T, [2, 3]);

        assert_eq!(a.shape(), &[2, 3]);
        assert_eq!(a.stride(), &[3, 1]);
        assert!(a.flatiter().all(|x| x == 3 as T));
        assert!(a.is_contiguous());
        assert_eq!(a.has_uniform_stride(), Some(1));
    }
);

#[test]
fn test_full_bool() {
    let a: NdArray<bool> = NdArray::full(true, vec![3, 5, 3]);
    assert_eq!(a.shape(), &[3, 5, 3]);
    assert_eq!(a.stride(), &[15, 3, 1]);
    assert!(a.flatiter().all(|x| x == true));
    assert!(a.is_contiguous());
    assert_eq!(a.has_uniform_stride(), Some(1));
}

test_for_all_numeric_dtypes!(
    test_ones, {
        let a = NdArray::<T>::ones([3, 5, 3]);

        assert_eq!(a.shape(), &[3, 5, 3]);
        assert_eq!(a.stride(), &[15, 3, 1]);
        assert!(a.flatiter().all(|x| x == 1 as T));
        assert!(a.is_contiguous());
        assert_eq!(a.has_uniform_stride(), Some(1));
    }
);


test_for_all_numeric_dtypes!(
    test_zeros, {
        let a = NdArray::<T>::zeros([3, 5, 3]);

        assert_eq!(a.shape(), &[3, 5, 3]);
        assert_eq!(a.stride(), &[15, 3, 1]);
        assert!(a.flatiter().all(|x| x == 0 as T));
        assert!(a.is_contiguous());
        assert_eq!(a.has_uniform_stride(), Some(1));
    }
);

#[test]
fn ones_bool() {
    let a: NdArray<bool> = NdArray::ones(vec![3, 5, 3]);
    assert_eq!(a.shape(), &[3, 5, 3]);
    assert_eq!(a.stride(), &[15, 3, 1]);
    assert!(a.flatiter().all(|x| x == true));
    assert!(a.is_contiguous());
    assert_eq!(a.has_uniform_stride(), Some(1));
}

#[test]
fn zeroes_bool() {
    let a: NdArray<bool> = NdArray::zeros(vec![3, 5, 3]);
    assert_eq!(a.shape(), &[3, 5, 3]);
    assert_eq!(a.stride(), &[15, 3, 1]);
    assert!(a.flatiter().all(|x| x == false));
    assert!(a.is_contiguous());
    assert_eq!(a.has_uniform_stride(), Some(1));
}

#[test]
fn random_normal_f32() {
    let a: NdArray<f32> = NdArray::randn(vec![3, 5, 3]);
    assert_eq!(a.shape(), &[3, 5, 3]);
    assert!(!a.is_view());
    assert!(a.is_contiguous());
    assert_eq!(a.has_uniform_stride(), Some(1));
}

#[test]
fn random_normal_f64() {
    let a: NdArray<f64> = NdArray::randn(vec![3, 5, 3]);
    let _: Vec<_> = a.flatiter().collect();
    assert_eq!(a.shape(), &[3, 5, 3]);
    assert!(!a.is_view());
    assert!(a.is_contiguous());
    assert_eq!(a.has_uniform_stride(), Some(1));
}

#[test]
fn random_uniform_f64() {
    let a: NdArray<f64> = NdArray::rand(vec![2, 3]);
    let _: Vec<_> = a.flatiter().collect();
    assert_eq!(a.shape(), &[2, 3]);
    assert!(!a.is_view());
    assert!(a.is_contiguous());
    assert_eq!(a.has_uniform_stride(), Some(1));
}

#[test]
fn random_uniform_f32() {
    let a: NdArray<f32> = NdArray::rand(vec![2, 3, 6]);
    let _: Vec<_> = a.flatiter().collect();
    assert_eq!(a.shape(), &[2, 3, 6]);
    assert!(!a.is_view());
    assert!(a.is_contiguous());
    assert_eq!(a.has_uniform_stride(), Some(1));
}


test_for_all_numeric_dtypes!(
    test_scalar, {
        let a = NdArray::scalar(5 as T);
        let _: Vec<_> = a.flatiter().collect();

        assert_eq!(a.shape(), &[]);
        assert!(!a.is_view());
        assert!(a.is_contiguous());
        assert_eq!(a.has_uniform_stride(), Some(0));
    }
);

test_for_all_numeric_dtypes!(
    test_arange, {
        let a = NdArray::<T>::arange(0 as T, 1 as T);
        let expected = NdArray::new([0]).astype::<T>();

        assert_eq!(a, expected);
        assert_eq!(a.shape(), &[1]);
        assert!(!a.is_view());
        assert!(a.is_contiguous());
        assert_eq!(a.has_uniform_stride(), Some(1));

        let b = NdArray::<T>::arange(8 as T, 15 as T);
        let expected = NdArray::new([8, 9, 10, 11, 12, 13, 14]).astype::<T>();
        assert_eq!(b, expected);
    }
);

test_for_all_numeric_dtypes!(
    test_arange_with_step, {
        let a = NdArray::<T>::arange_with_step(0 as T, 1 as T, 2 as T);
        let expected = NdArray::new([0]).astype::<T>();

        assert_eq!(a, expected);
        assert_eq!(a.shape(), &[1]);
        assert!(!a.is_view());
        assert!(a.is_contiguous());
        assert_eq!(a.has_uniform_stride(), Some(1));

        let b = NdArray::<T>::arange_with_step(8 as T, 15 as T, 2 as T);
        let expected = NdArray::new([8, 10, 12, 14]).astype::<T>();
        assert_eq!(b, expected);
    }
);

test_for_signed_dtypes!(
    test_arange_with_negative_step, {
        let a = NdArray::<T>::arange_with_step(15 as T, 8 as T, -3 as T);
        let expected = NdArray::new([15, 12, 9]).astype::<T>();
        assert_eq!(a, expected);

        let a = NdArray::<T>::arange_with_step(21 as T, -48 as T, -7 as T);
        let expected = NdArray::new([21, 14, 7, 0, -7, -14, -21, -28, -35, -42]).astype::<T>();
        assert_eq!(a, expected);
    }
);

test_for_float_dtypes!(
    test_float_arange, {
        let a = NdArray::<T>::arange(1.5, 6.4);
        let expected = NdArray::<T>::new([1.5, 2.5, 3.5, 4.5, 5.5]);
        assert_almost_eq!(a, expected);

        let b = NdArray::<T>::arange(-5.3 as T, 1.4 as T);
        let expected = NdArray::<T>::new([-5.3, -4.3, -3.3, -2.3, -1.3, 0.3, 1.3]);
        assert_almost_eq!(b, expected);
    }
);

test_for_float_dtypes!(
    test_linspace, {
        let a = NdArray::<T>::linspace_exclusive(1.0, 3.0, 5);
        let expected = NdArray::<T>::new([1.0, 1.4, 1.8, 2.2, 2.6]);
        assert_almost_eq!(a, expected);

        let a = NdArray::<T>::linspace(1.0, 3.0, 5);
        let expected = NdArray::<T>::new([1.0, 1.5, 2.0, 2.5, 3.0]);
        assert_almost_eq!(a, expected);
        
        let a = NdArray::<T>::linspace(1.0, 3.0, 1);
        let expected = NdArray::<T>::new([1.0]);
        assert_almost_eq!(a, expected);
    }
);
