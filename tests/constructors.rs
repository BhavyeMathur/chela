use chela::*;
use paste::paste;

test_for_all_numeric_types!(
    test_full, {
        let a = Tensor::full(3 as T, [2, 3]);

        assert_eq!(a.shape(), &[2, 3]);
        assert_eq!(a.stride(), &[3, 1]);
        assert!(a.flatiter().all(|x| x == 3 as T));
        assert!(a.is_contiguous());
        assert_eq!(a.has_uniform_stride(), Some(1));
    }
);

#[test]
fn test_full_bool() {
    let a: Tensor<bool> = Tensor::full(true, vec![3, 5, 3]);
    assert_eq!(a.shape(), &[3, 5, 3]);
    assert_eq!(a.stride(), &[15, 3, 1]);
    assert!(a.flatiter().all(|x| x == true));
    assert!(a.is_contiguous());
    assert_eq!(a.has_uniform_stride(), Some(1));
}

test_for_all_numeric_types!(
    test_ones, {
        let a = Tensor::<T>::ones([3, 5, 3]);

        assert_eq!(a.shape(), &[3, 5, 3]);
        assert_eq!(a.stride(), &[15, 3, 1]);
        assert!(a.flatiter().all(|x| x == 1 as T));
        assert!(a.is_contiguous());
        assert_eq!(a.has_uniform_stride(), Some(1));
    }
);


test_for_all_numeric_types!(
    test_zeros, {
        let a = Tensor::<T>::zeros([3, 5, 3]);

        assert_eq!(a.shape(), &[3, 5, 3]);
        assert_eq!(a.stride(), &[15, 3, 1]);
        assert!(a.flatiter().all(|x| x == 0 as T));
        assert!(a.is_contiguous());
        assert_eq!(a.has_uniform_stride(), Some(1));
    }
);

#[test]
fn ones_bool() {
    let a: Tensor<bool> = Tensor::ones(vec![3, 5, 3]);
    assert_eq!(a.shape(), &[3, 5, 3]);
    assert_eq!(a.stride(), &[15, 3, 1]);
    assert!(a.flatiter().all(|x| x == true));
    assert!(a.is_contiguous());
    assert_eq!(a.has_uniform_stride(), Some(1));
}

#[test]
fn zeroes_bool() {
    let a: Tensor<bool> = Tensor::zeros(vec![3, 5, 3]);
    assert_eq!(a.shape(), &[3, 5, 3]);
    assert_eq!(a.stride(), &[15, 3, 1]);
    assert!(a.flatiter().all(|x| x == false));
    assert!(a.is_contiguous());
    assert_eq!(a.has_uniform_stride(), Some(1));
}

#[test]
fn random_normal_f32() {
    let a: Tensor<f32> = Tensor::randn(vec![3, 5, 3]);
    assert_eq!(a.shape(), &[3, 5, 3]);
    assert!(!a.is_view());
    assert!(a.is_contiguous());
    assert_eq!(a.has_uniform_stride(), Some(1));
}

#[test]
fn random_normal_f64() {
    let a: Tensor<f64> = Tensor::randn(vec![3, 5, 3]);
    let _: Vec<_> = a.flatiter().collect();
    assert_eq!(a.shape(), &[3, 5, 3]);
    assert!(!a.is_view());
    assert!(a.is_contiguous());
    assert_eq!(a.has_uniform_stride(), Some(1));
}

#[test]
fn random_uniform_f64() {
    let a: Tensor<f64> = Tensor::rand(vec![2, 3]);
    let _: Vec<_> = a.flatiter().collect();
    assert_eq!(a.shape(), &[2, 3]);
    assert!(!a.is_view());
    assert!(a.is_contiguous());
    assert_eq!(a.has_uniform_stride(), Some(1));
}

#[test]
fn random_uniform_f32() {
    let a: Tensor<f32> = Tensor::rand(vec![2, 3, 6]);
    let _: Vec<_> = a.flatiter().collect();
    assert_eq!(a.shape(), &[2, 3, 6]);
    assert!(!a.is_view());
    assert!(a.is_contiguous());
    assert_eq!(a.has_uniform_stride(), Some(1));
}


test_for_all_numeric_types!(
    test_scalar, {
        let a = Tensor::scalar(5 as T);
        let _: Vec<_> = a.flatiter().collect();

        assert_eq!(a.shape(), &[]);
        assert!(!a.is_view());
        assert!(a.is_contiguous());
        assert_eq!(a.has_uniform_stride(), Some(0));
    }
);

test_for_all_numeric_types!(
    test_arange, {
        let a = Tensor::<T>::arange(0 as T, 1 as T);
        let expected = Tensor::from([0]).astype::<T>();

        assert_eq!(a, expected);
        assert_eq!(a.shape(), &[1]);
        assert!(!a.is_view());
        assert!(a.is_contiguous());
        assert_eq!(a.has_uniform_stride(), Some(1));

        let b = Tensor::<T>::arange(8 as T, 15 as T);
        let expected = Tensor::from([8, 9, 10, 11, 12, 13, 14]).astype::<T>();
        assert_eq!(b, expected);
    }
);

// TODO implement assert_almost_eq!
// test_for_all_float_types!(
//     test_float_arange, {
//         let a = Tensor::<T>::arange(1.5, 6.4);
//         let expected = Tensor::from([1.5, 2.5, 3.5, 4.5, 5.5]).astype::<T>();
//         assert_almost_eq!(a, expected);
//
//         let b = Tensor::<T>::arange(-5.3 as T, 1.4 as T);
//         let expected = Tensor::from([-5.3, -4.3, -3.3, -2.3, -1.3, 0.3, 1.3]).astype::<T>();
//         assert_almost_eq!(b, expected);
//     }
// );
