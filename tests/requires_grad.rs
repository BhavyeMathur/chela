use chela::*;
use paste::paste;

#[test]
fn test_constructor_requires_grad() {
    let a = Tensor::from([1, 2, 3]);
    assert!(!a.requires_grad());

    let a = Tensor::full(5i32, [1, 2, 3]);
    assert!(!a.requires_grad());

    let a = Tensor::linspace(0f32, 5.0, 2);
    assert!(!a.requires_grad());

    let a = Tensor::linspace_exclusive(0f64, 5.0, 2);
    assert!(!a.requires_grad());

    let a = Tensor::arange(0, 12);
    assert!(!a.requires_grad());

    let a = Tensor::arange_with_step(0, -12, -3);
    assert!(!a.requires_grad());

    let mut a = Tensor::full(5i32, [1, 2, 3]);
    a.set_requires_grad(false);
    assert!(!a.requires_grad());

    let mut a = Tensor::full(5i32, [1, 2, 3]);
    a.set_requires_grad(true);
    assert!(a.requires_grad());
}

#[test]
fn test_iter_requires_grad() {
    for requires_grad in [false, true] {
        let mut a = Tensor::<i64>::ones([1, 2, 3]);
        a.set_requires_grad(requires_grad);
        
        for b in a.iter_along(Axis(-1)) {
            assert_eq!(b.requires_grad(), requires_grad);
        }
        
        let b = a.slice(s![.., 0, 0..2]);
        assert_eq!(b.requires_grad(), requires_grad);

        let b = a.slice_along(Axis(-2), 0);
        assert_eq!(b.requires_grad(), requires_grad);
    }
}

#[test]
fn test_reshape_requires_grad() {
    for requires_grad in [false, true] {
        let mut a = Tensor::<u32>::ones([1, 2, 3]);
        a.set_requires_grad(requires_grad);
        
        let b = a.reshape([6, 1]);
        assert_eq!(b.requires_grad(), requires_grad);

        let b = a.flatten();
        assert_eq!(b.requires_grad(), requires_grad);

        let b = a.diagonal();
        assert_eq!(b.requires_grad(), requires_grad);

        let b = a.clone();
        assert_eq!(b.requires_grad(), requires_grad);

        let b = a.view();
        assert_eq!(b.requires_grad(), requires_grad);

        let b = a.squeeze();
        assert_eq!(b.requires_grad(), requires_grad);

        let b = a.unsqueeze(Axis(-1));
        assert_eq!(b.requires_grad(), requires_grad);
    }
}

test_for_common_numeric_dtypes!(
 test_reduce_requires_grad, {
        for requires_grad in [false, true] {
            let mut a = Tensor::<f32>::zeros([4, 4, 2]);
            a.set_requires_grad(requires_grad);

            let b = a.sum();
            assert_eq!(b.requires_grad(), requires_grad);

            let b = a.sum_along(0);
            assert_eq!(b.requires_grad(), requires_grad);

            let b = a.min();
            assert_eq!(b.requires_grad(), requires_grad);

            let b = a.min_along(0);
            assert_eq!(b.requires_grad(), requires_grad);

            let b = a.max();
            assert_eq!(b.requires_grad(), requires_grad);

            let b = a.max_along(0);
            assert_eq!(b.requires_grad(), requires_grad);

            let b = a.mean();
            assert_eq!(b.requires_grad(), requires_grad);

            let b = a.mean_along(0);
            assert_eq!(b.requires_grad(), requires_grad);

            let b = a.product();
            assert_eq!(b.requires_grad(), requires_grad);

            let b = a.product_along(0);
            assert_eq!(b.requires_grad(), requires_grad);

            let b = a.min_magnitude();
            assert_eq!(b.requires_grad(), requires_grad);

            let b = a.min_magnitude_along(0);
            assert_eq!(b.requires_grad(), requires_grad);

            let b = a.max_magnitude();
            assert_eq!(b.requires_grad(), requires_grad);

            let b = a.max_magnitude_along(0);
            assert_eq!(b.requires_grad(), requires_grad);
        }
    }
);
