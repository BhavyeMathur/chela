use chela::*;
use num::{NumCast, One, Zero};
use paste::paste;


test_for_all_numeric_dtypes!(
    test_add, {
        for n in 1..23 {
            let tensor1 = NdArray::arange(0, n).squeeze().astype::<T>();
            let tensor2 = NdArray::arange(n, n * 2).squeeze().astype::<T>();

            let correct = NdArray::arange_with_step(n, 2 * n + n, 2).squeeze().astype::<T>();
            assert_eq!(&tensor1 + &tensor2, correct);
            assert_eq!(&tensor1 + tensor2, correct);

            let tensor2 = NdArray::arange(n, n * 2).squeeze().astype::<T>();
            assert_eq!(tensor1 + &tensor2, correct);

            let tensor1 = NdArray::arange(0, n).squeeze().astype::<T>();
            assert_eq!(tensor1 + tensor2, correct);

            let tensor1 = NdArray::arange(0, n).squeeze().astype::<T>();
            let tensor2 = NdArray::arange(n, n * 2).squeeze().astype::<T>();
            assert_eq!(&tensor2 + &tensor1, correct);
            assert_eq!(&tensor2 + tensor1, correct);

            let tensor1 = NdArray::arange(0, n).squeeze().astype::<T>();
            assert_eq!(tensor2 + &tensor1, correct);

            let tensor2 = NdArray::arange(n, n * 2).squeeze().astype::<T>();
            let tensor1 = NdArray::arange(0, n).squeeze().astype::<T>();
            assert_eq!(tensor2 + tensor1, correct);
        }
    }
);

test_for_common_numeric_dtypes!(
    test_add_non_contiguous, {
        let ten = NumCast::from(10).unwrap();

        for n in 1..23 {
            // inner strides: [k], [1]
            let tensor1 = NdArray::<T>::randint([n, 2], T::zero(), ten);
            let tensor1 = tensor1.slice_along(Axis(-1), 0);

            let tensor2 = NdArray::<T>::randint([n], T::zero(), ten);

            let correct: Vec<T> = tensor1.flatiter().zip(tensor2.flatiter()).map(|(lhs, rhs)| lhs + rhs).collect();
            let correct = NdArray::new(correct);

            assert_eq!(&tensor1 + &tensor2, correct);
            assert_eq!(&tensor2 + &tensor1, correct);

            // inner strides: [k], [j]
            let tensor1 = NdArray::<T>::randint([n, 2], T::zero(), ten);
            let tensor1 = tensor1.slice_along(Axis(-1), 0);

            let tensor2 = NdArray::<T>::randint([n, 3], T::zero(), ten);
            let tensor2 = tensor2.slice_along(Axis(-1), 0);

            let correct: Vec<T> = tensor1.flatiter().zip(tensor2.flatiter()).map(|(lhs, rhs)| lhs + rhs).collect();
            let correct = NdArray::new(correct);

            assert_eq!(&tensor1 + &tensor2, correct);
            assert_eq!(&tensor2 + &tensor1, correct);

            // inner strides: [1], [non-unif]
            let tensor1 = NdArray::<T>::randint([n, 2], T::zero(), ten);

            let tensor2 = NdArray::<T>::randint([n, 3], T::zero(), ten);
            let tensor2 = tensor2.slice_along(Axis(-1), 0..2);

            let correct: Vec<T> = tensor1.flatiter().zip(tensor2.flatiter()).map(|(lhs, rhs)| lhs + rhs).collect();
            let correct = NdArray::new(correct).reshape([n, 2]);

            assert_eq!(&tensor1 + &tensor2, correct);
            assert_eq!(&tensor2 + &tensor1, correct);

            // inner strides: [k], [non-unif]
            let tensor1 = NdArray::<T>::randint([n, 2, 2], T::zero(), ten);
            let tensor1 = tensor1.slice_along(Axis(-1), 0);

            let tensor2 = NdArray::<T>::randint([n, 3], T::zero(), ten);
            let tensor2 = tensor2.slice_along(Axis(-1), 0..2);

            let correct: Vec<T> = tensor1.flatiter().zip(tensor2.flatiter()).map(|(lhs, rhs)| lhs + rhs).collect();
            let correct = NdArray::new(correct).reshape([n, 2]);

            assert_eq!(&tensor1 + &tensor2, correct);
            assert_eq!(&tensor2 + &tensor1, correct);

            // inner strides: [non-unif], [non-unif]
            let tensor1 = NdArray::<T>::randint([n, 3], T::zero(), ten);
            let tensor1 = tensor1.slice_along(Axis(-1), 0..2);

            let tensor2 = NdArray::<T>::randint([n, 5], T::zero(), ten);
            let tensor2 = tensor2.slice_along(Axis(-1), 2..4);

            let correct: Vec<T> = tensor1.flatiter().zip(tensor2.flatiter()).map(|(lhs, rhs)| lhs + rhs).collect();
            let correct = NdArray::new(correct).reshape([n, 2]);

            assert_eq!(&tensor1 + &tensor2, correct);
            assert_eq!(&tensor2 + &tensor1, correct);
        }
    }
);

test_for_common_numeric_dtypes!(
    test_add_scalar, {
        let five = NumCast::from(5).unwrap();

        // inner strides: [0], [0]
        let tensor1 = NdArray::scalar(10).astype::<T>();
        let tensor2 = NdArray::scalar(5).astype::<T>();
        let correct = NdArray::scalar(15).astype::<T>();
        assert_almost_eq!(&tensor1 + &tensor2, correct);
        assert_almost_eq!(&tensor1 + five, correct);
        assert_almost_eq!(&tensor2 + &tensor1, correct);

        for n in 1..23 {
            // inner strides: [1], [0]
            let tensor1 = NdArray::<T>::randint([n], T::one(), five);

            let correct: Vec<T> = tensor1.flatiter().map(|lhs| lhs + five).collect();
            let correct = NdArray::new(correct);

            assert_almost_eq!(&tensor1 + &tensor2, correct);
            assert_almost_eq!(&tensor1 + five, correct);
            assert_almost_eq!(&tensor2 + &tensor1, correct);

            // inner strides: [k], [0]
            let tensor1 = NdArray::<T>::randint([n, 2], T::one(), five);
            let tensor1 = tensor1.slice_along(Axis(-1), 0);

            let correct: Vec<T> = tensor1.flatiter().map(|lhs| lhs + five).collect();
            let correct = NdArray::new(correct);

            assert_almost_eq!(&tensor1 + &tensor2, correct);
            assert_almost_eq!(&tensor1 + five, correct);
            assert_almost_eq!(&tensor2 + &tensor1, correct);

            // inner strides: [non-unif], [0]
            let tensor1 = NdArray::<T>::randint([n, 3], T::one(), five);
            let tensor1 = tensor1.slice_along(Axis(-1), 0..2);

            let correct: Vec<T> = tensor1.flatiter().map(|lhs| lhs + five).collect();
            let correct = NdArray::new(correct).reshape([n, 2]);

            assert_almost_eq!(&tensor1 + &tensor2, correct);
            assert_almost_eq!(&tensor1 + five, correct);
            assert_almost_eq!(&tensor2 + &tensor1, correct);
        }
    }
);

test_for_signed_dtypes!(
    test_sub, {
        for n in 1..23 {
            let tensor1 = NdArray::arange(0, n).squeeze().astype::<T>();
            let tensor2 = NdArray::arange(n, n * 2).squeeze().astype::<T>();

            let correct: Vec<T> = tensor1.flatiter().zip(tensor2.flatiter()).map(|(lhs, rhs)| lhs - rhs).collect();
            let correct = NdArray::new(correct).squeeze();
            assert_eq!(&tensor1 - &tensor2, correct);
            assert_eq!(&tensor1 - tensor2, correct);

            let tensor2 = NdArray::arange(n, n * 2).squeeze().astype::<T>();
            assert_eq!(tensor1 - &tensor2, correct);

            let tensor1 = NdArray::arange(0, n).squeeze().astype::<T>();
            assert_eq!(tensor1 - tensor2, correct);

            let correct = -correct;
            let tensor1 = NdArray::arange(0, n).squeeze().astype::<T>();
            let tensor2 = NdArray::arange(n, n * 2).squeeze().astype::<T>();
            assert_eq!(&tensor2 - &tensor1, correct);
            assert_eq!(&tensor2 - tensor1, correct);

            let tensor1 = NdArray::arange(0, n).squeeze().astype::<T>();
            assert_eq!(tensor2 - &tensor1, correct);

            let tensor2 = NdArray::arange(n, n * 2).squeeze().astype::<T>();
            let tensor1 = NdArray::arange(0, n).squeeze().astype::<T>();
            assert_eq!(tensor2 - tensor1, correct);
        }
    }
);

test_for_signed_dtypes!(
    test_sub_non_contiguous, {
        let ten = NumCast::from(10).unwrap();

        for n in 1..23 {
            // inner strides: [k], [1]
            let tensor1 = NdArray::<T>::randint([n, 2], T::zero(), ten);
            let tensor1 = tensor1.slice_along(Axis(-1), 0);

            let tensor2 = NdArray::<T>::randint([n], T::zero(), ten);

            let correct: Vec<T> = tensor1.flatiter().zip(tensor2.flatiter()).map(|(lhs, rhs)| lhs - rhs).collect();
            let correct = NdArray::new(correct);

            assert_eq!(&tensor1 - &tensor2, correct);
            assert_eq!(&tensor2 - &tensor1, -correct);

            // inner strides: [k], [j]
            let tensor1 = NdArray::<T>::randint([n, 2], T::zero(), ten);
            let tensor1 = tensor1.slice_along(Axis(-1), 0);

            let tensor2 = NdArray::<T>::randint([n, 3], T::zero(), ten);
            let tensor2 = tensor2.slice_along(Axis(-1), 0);

            let correct: Vec<T> = tensor1.flatiter().zip(tensor2.flatiter()).map(|(lhs, rhs)| lhs - rhs).collect();
            let correct = NdArray::new(correct);

            assert_eq!(&tensor1 - &tensor2, correct);
            assert_eq!(&tensor2 - &tensor1, -correct);

            // inner strides: [1], [non-unif]
            let tensor1 = NdArray::<T>::randint([n, 2], T::zero(), ten);

            let tensor2 = NdArray::<T>::randint([n, 3], T::zero(), ten);
            let tensor2 = tensor2.slice_along(Axis(-1), 0..2);

            let correct: Vec<T> = tensor1.flatiter().zip(tensor2.flatiter()).map(|(lhs, rhs)| lhs - rhs).collect();
            let correct = NdArray::new(correct).reshape([n, 2]);

            assert_eq!(&tensor1 - &tensor2, correct);
            assert_eq!(&tensor2 - &tensor1, -correct);

            // inner strides: [k], [non-unif]
            let tensor1 = NdArray::<T>::randint([n, 2, 2], T::zero(), ten);
            let tensor1 = tensor1.slice_along(Axis(-1), 0);

            let tensor2 = NdArray::<T>::randint([n, 3], T::zero(), ten);
            let tensor2 = tensor2.slice_along(Axis(-1), 0..2);

            let correct: Vec<T> = tensor1.flatiter().zip(tensor2.flatiter()).map(|(lhs, rhs)| lhs - rhs).collect();
            let correct = NdArray::new(correct).reshape([n, 2]);

            assert_eq!(&tensor1 - &tensor2, correct);
            assert_eq!(&tensor2 - &tensor1, -correct);

            // inner strides: [non-unif], [non-unif]
            let tensor1 = NdArray::<T>::randint([n, 3], T::zero(), ten);
            let tensor1 = tensor1.slice_along(Axis(-1), 0..2);

            let tensor2 = NdArray::<T>::randint([n, 5], T::zero(), ten);
            let tensor2 = tensor2.slice_along(Axis(-1), 2..4);

            let correct: Vec<T> = tensor1.flatiter().zip(tensor2.flatiter()).map(|(lhs, rhs)| lhs - rhs).collect();
            let correct = NdArray::new(correct).reshape([n, 2]);

            assert_eq!(&tensor1 - &tensor2, correct);
            assert_eq!(&tensor2 - &tensor1, -correct);
        }
    }
);

test_for_signed_dtypes!(
    test_sub_scalar, {
        let five = NumCast::from(5).unwrap();

        // inner strides: [0], [0]
        let tensor1 = NdArray::scalar(10).astype::<T>();
        let tensor2 = NdArray::scalar(5).astype::<T>();
        let correct = NdArray::scalar(5).astype::<T>();
        assert_almost_eq!(&tensor1 - &tensor2, correct);
        assert_almost_eq!(&tensor1 - five, correct);
        assert_almost_eq!(&tensor2 - &tensor1, -&correct);

        for n in 1..23 {
            // inner strides: [1], [0]
            let tensor1 = NdArray::<T>::randint([n], T::one(), five);

            let correct: Vec<T> = tensor1.flatiter().map(|lhs| lhs - five).collect();
            let correct = NdArray::new(correct);

            assert_almost_eq!(&tensor1 - &tensor2, correct);
            assert_almost_eq!(&tensor1 - five, correct);
            assert_almost_eq!(&tensor2 - &tensor1, -&correct);

            // inner strides: [k], [0]
            let tensor1 = NdArray::<T>::randint([n, 2], T::one(), five);
            let tensor1 = tensor1.slice_along(Axis(-1), 0);

            let correct: Vec<T> = tensor1.flatiter().map(|lhs| lhs - five).collect();
            let correct = NdArray::new(correct);

            assert_almost_eq!(&tensor1 - &tensor2, correct);
            assert_almost_eq!(&tensor1 - five, correct);
            assert_almost_eq!(&tensor2 - &tensor1, -&correct);

            // inner strides: [non-unif], [0]
            let tensor1 = NdArray::<T>::randint([n, 3], T::one(), five);
            let tensor1 = tensor1.slice_along(Axis(-1), 0..2);

            let correct: Vec<T> = tensor1.flatiter().map(|lhs| lhs - five).collect();
            let correct = NdArray::new(correct).reshape([n, 2]);

            assert_almost_eq!(&tensor1 - &tensor2, correct);
            assert_almost_eq!(&tensor1 - five, correct);
            assert_almost_eq!(&tensor2 - &tensor1, -&correct);
        }
    }
);

test_for_common_numeric_dtypes!(
    test_mul, {
        for n in 1..23 {
            let tensor1 = NdArray::arange(0, n).squeeze().astype::<T>();
            let tensor2 = NdArray::arange(n, n * 2).squeeze().astype::<T>();

            let correct: Vec<T> = tensor1.flatiter().zip(tensor2.flatiter()).map(|(lhs, rhs)| lhs * rhs).collect();
            let correct = NdArray::new(correct).squeeze();
            assert_eq!(&tensor1 * &tensor2, correct);
            assert_eq!(&tensor1 * tensor2, correct);

            let tensor2 = NdArray::arange(n, n * 2).squeeze().astype::<T>();
            assert_eq!(tensor1 * &tensor2, correct);

            let tensor1 = NdArray::arange(0, n).squeeze().astype::<T>();
            assert_eq!(tensor1 * tensor2, correct);

            let tensor1 = NdArray::arange(0, n).squeeze().astype::<T>();
            let tensor2 = NdArray::arange(n, n * 2).squeeze().astype::<T>();
            assert_eq!(&tensor2 * &tensor1, correct);
            assert_eq!(&tensor2 * tensor1, correct);

            let tensor1 = NdArray::arange(0, n).squeeze().astype::<T>();
            assert_eq!(tensor2 * &tensor1, correct);

            let tensor2 = NdArray::arange(n, n * 2).squeeze().astype::<T>();
            let tensor1 = NdArray::arange(0, n).squeeze().astype::<T>();
            assert_eq!(tensor2 * tensor1, correct);
        }
    }
);

test_for_common_numeric_dtypes!(
    test_mul_non_contiguous, {
        let ten = NumCast::from(10).unwrap();

        for n in 1..23 {
            // inner strides: [k], [1]
            let tensor1 = NdArray::<T>::randint([n, 2], T::zero(), ten);
            let tensor1 = tensor1.slice_along(Axis(-1), 0);

            let tensor2 = NdArray::<T>::randint([n], T::zero(), ten);

            let correct: Vec<T> = tensor1.flatiter().zip(tensor2.flatiter()).map(|(lhs, rhs)| lhs * rhs).collect();
            let correct = NdArray::new(correct);

            assert_eq!(&tensor1 * &tensor2, correct);
            assert_eq!(&tensor2 * &tensor1, correct);

            // inner strides: [k], [j]
            let tensor1 = NdArray::<T>::randint([n, 2], T::zero(), ten);
            let tensor1 = tensor1.slice_along(Axis(-1), 0);

            let tensor2 = NdArray::<T>::randint([n, 3], T::zero(), ten);
            let tensor2 = tensor2.slice_along(Axis(-1), 0);

            let correct: Vec<T> = tensor1.flatiter().zip(tensor2.flatiter()).map(|(lhs, rhs)| lhs * rhs).collect();
            let correct = NdArray::new(correct);

            assert_eq!(&tensor1 * &tensor2, correct);
            assert_eq!(&tensor2 * &tensor1, correct);

            // inner strides: [1], [non-unif]
            let tensor1 = NdArray::<T>::randint([n, 2], T::zero(), ten);

            let tensor2 = NdArray::<T>::randint([n, 3], T::zero(), ten);
            let tensor2 = tensor2.slice_along(Axis(-1), 0..2);

            let correct: Vec<T> = tensor1.flatiter().zip(tensor2.flatiter()).map(|(lhs, rhs)| lhs * rhs).collect();
            let correct = NdArray::new(correct).reshape([n, 2]);

            assert_eq!(&tensor1 * &tensor2, correct);
            assert_eq!(&tensor2 * &tensor1, correct);

            // inner strides: [k], [non-unif]
            let tensor1 = NdArray::<T>::randint([n, 2, 2], T::zero(), ten);
            let tensor1 = tensor1.slice_along(Axis(-1), 0);

            let tensor2 = NdArray::<T>::randint([n, 3], T::zero(), ten);
            let tensor2 = tensor2.slice_along(Axis(-1), 0..2);

            let correct: Vec<T> = tensor1.flatiter().zip(tensor2.flatiter()).map(|(lhs, rhs)| lhs * rhs).collect();
            let correct = NdArray::new(correct).reshape([n, 2]);

            assert_eq!(&tensor1 * &tensor2, correct);
            assert_eq!(&tensor2 * &tensor1, correct);

            // inner strides: [non-unif], [non-unif]
            let tensor1 = NdArray::<T>::randint([n, 3], T::zero(), ten);
            let tensor1 = tensor1.slice_along(Axis(-1), 0..2);

            let tensor2 = NdArray::<T>::randint([n, 5], T::zero(), ten);
            let tensor2 = tensor2.slice_along(Axis(-1), 2..4);

            let correct: Vec<T> = tensor1.flatiter().zip(tensor2.flatiter()).map(|(lhs, rhs)| lhs * rhs).collect();
            let correct = NdArray::new(correct).reshape([n, 2]);

            assert_eq!(&tensor1 * &tensor2, correct);
            assert_eq!(&tensor2 * &tensor1, correct);
        }
    }
);

test_for_common_numeric_dtypes!(
    test_mul_scalar, {
        let five = NumCast::from(5).unwrap();

        // inner strides: [0], [0]
        let tensor1 = NdArray::scalar(10).astype::<T>();
        let tensor2 = NdArray::scalar(5).astype::<T>();
        let correct = NdArray::scalar(50).astype::<T>();
        assert_almost_eq!(&tensor1 * &tensor2, correct);
        assert_almost_eq!(&tensor1 * five, correct);
        assert_almost_eq!(&tensor2 * &tensor1, correct);

        for n in 1..23 {
            // inner strides: [1], [0]
            let tensor1 = NdArray::<T>::randint([n], T::one(), five);

            let correct: Vec<T> = tensor1.flatiter().map(|lhs| lhs * five).collect();
            let correct = NdArray::new(correct);

            assert_almost_eq!(&tensor1 * &tensor2, correct);
            assert_almost_eq!(&tensor1 * five, correct);
            assert_almost_eq!(&tensor2 * &tensor1, correct);

            // inner strides: [k], [0]
            let tensor1 = NdArray::<T>::randint([n, 2], T::one(), five);
            let tensor1 = tensor1.slice_along(Axis(-1), 0);

            let correct: Vec<T> = tensor1.flatiter().map(|lhs| lhs * five).collect();
            let correct = NdArray::new(correct);

            assert_almost_eq!(&tensor1 * &tensor2, correct);
            assert_almost_eq!(&tensor1 * five, correct);
            assert_almost_eq!(&tensor2 * &tensor1, correct);

            // inner strides: [non-unif], [0]
            let tensor1 = NdArray::<T>::randint([n, 3], T::one(), five);
            let tensor1 = tensor1.slice_along(Axis(-1), 0..2);

            let correct: Vec<T> = tensor1.flatiter().map(|lhs| lhs * five).collect();
            let correct = NdArray::new(correct).reshape([n, 2]);

            assert_almost_eq!(&tensor1 * &tensor2, correct);
            assert_almost_eq!(&tensor1 * five, correct);
            assert_almost_eq!(&tensor2 * &tensor1, correct);
        }
    }
);

test_for_float_dtypes!(
    test_div, {
        let one = NdArray::scalar(T::one());

        for n in 1..23 {
            let tensor1 = NdArray::arange(0, n).squeeze().astype::<T>();
            let tensor2 = NdArray::arange(n, n * 2).squeeze().astype::<T>();

            let correct: Vec<T> = tensor1.flatiter().zip(tensor2.flatiter()).map(|(lhs, rhs)| lhs / rhs).collect();
            let correct = NdArray::new(correct).squeeze();

            assert_almost_eq!(&tensor1 / &tensor2, correct);
            assert_almost_eq!(&tensor2 / &tensor1, &one / &correct);
        }
    }
);

test_for_float_dtypes!(
    test_div_non_contiguous, {
        let ten = NumCast::from(10).unwrap();
        let one = NdArray::scalar(T::one());

        for n in 1..23 {
            // inner strides: [k], [1]
            let tensor1 = NdArray::<T>::randint([n, 2], T::one(), ten);
            let tensor1 = tensor1.slice_along(Axis(-1), 0);

            let tensor2 = NdArray::<T>::randint([n], T::one(), ten);

            let correct: Vec<T> = tensor1.flatiter().zip(tensor2.flatiter()).map(|(lhs, rhs)| lhs / rhs).collect();
            let correct = NdArray::new(correct);

            assert_almost_eq!(&tensor1 / &tensor2, correct);
            assert_almost_eq!(&tensor2 / &tensor1, &one / &correct);

            // inner strides: [k], [j]
            let tensor1 = NdArray::<T>::randint([n, 2], T::one(), ten);
            let tensor1 = tensor1.slice_along(Axis(-1), 0);

            let tensor2 = NdArray::<T>::randint([n, 3], T::one(), ten);
            let tensor2 = tensor2.slice_along(Axis(-1), 0);

            let correct: Vec<T> = tensor1.flatiter().zip(tensor2.flatiter()).map(|(lhs, rhs)| lhs / rhs).collect();
            let correct = NdArray::new(correct);

            assert_almost_eq!(&tensor1 / &tensor2, correct);
            assert_almost_eq!(&tensor2 / &tensor1, &one / &correct);

            // inner strides: [1], [non-unif]
            let tensor1 = NdArray::<T>::randint([n, 2], T::one(), ten);

            let tensor2 = NdArray::<T>::randint([n, 3], T::one(), ten);
            let tensor2 = tensor2.slice_along(Axis(-1), 0..2);

            let correct: Vec<T> = tensor1.flatiter().zip(tensor2.flatiter()).map(|(lhs, rhs)| lhs / rhs).collect();
            let correct = NdArray::new(correct).reshape([n, 2]);

            assert_almost_eq!(&tensor1 / &tensor2, correct);
            assert_almost_eq!(&tensor2 / &tensor1, &one / &correct);

            // inner strides: [k], [non-unif]
            let tensor1 = NdArray::<T>::randint([n, 2, 2], T::one(), ten);
            let tensor1 = tensor1.slice_along(Axis(-1), 0);

            let tensor2 = NdArray::<T>::randint([n, 3], T::one(), ten);
            let tensor2 = tensor2.slice_along(Axis(-1), 0..2);

            let correct: Vec<T> = tensor1.flatiter().zip(tensor2.flatiter()).map(|(lhs, rhs)| lhs / rhs).collect();
            let correct = NdArray::new(correct).reshape([n, 2]);

            assert_almost_eq!(&tensor1 / &tensor2, correct);
            assert_almost_eq!(&tensor2 / &tensor1, &one / &correct);

            // inner strides: [non-unif], [non-unif]
            let tensor1 = NdArray::<T>::randint([n, 3], T::one(), ten);
            let tensor1 = tensor1.slice_along(Axis(-1), 0..2);

            let tensor2 = NdArray::<T>::randint([n, 5], T::one(), ten);
            let tensor2 = tensor2.slice_along(Axis(-1), 2..4);

            let correct: Vec<T> = tensor1.flatiter().zip(tensor2.flatiter()).map(|(lhs, rhs)| lhs / rhs).collect();
            let correct = NdArray::new(correct).reshape([n, 2]);

            assert_almost_eq!(&tensor1 / &tensor2, correct);
            assert_almost_eq!(&tensor2 / &tensor1, &one / &correct);
        }
    }
);

test_for_float_dtypes!(
    test_div_scalar, {
        let five = NumCast::from(5).unwrap();
        let one = NdArray::scalar(T::one());

        // inner strides: [0], [0]
        let tensor1 = NdArray::scalar(10).astype::<T>();
        let tensor2 = NdArray::scalar(5).astype::<T>();
        let correct = NdArray::scalar(2).astype::<T>();
        assert_almost_eq!(&tensor1 / &tensor2, correct);
        assert_almost_eq!(&tensor1 / five, correct);
        assert_almost_eq!(&tensor2 / &tensor1, &one / &correct);

        for n in 1..23 {
            // inner strides: [1], [0]
            let tensor1 = NdArray::<T>::randint([n], T::one(), five);

            let correct: Vec<T> = tensor1.flatiter().map(|lhs| lhs / five).collect();
            let correct = NdArray::new(correct);

            assert_almost_eq!(&tensor1 / &tensor2, correct);
            assert_almost_eq!(&tensor1 / five, correct);
            assert_almost_eq!(&tensor2 / &tensor1, &one / &correct);

            // inner strides: [k], [0]
            let tensor1 = NdArray::<T>::randint([n, 2], T::one(), five);
            let tensor1 = tensor1.slice_along(Axis(-1), 0);

            let correct: Vec<T> = tensor1.flatiter().map(|lhs| lhs / five).collect();
            let correct = NdArray::new(correct);

            assert_almost_eq!(&tensor1 / &tensor2, correct);
            assert_almost_eq!(&tensor1 / five, correct);
            assert_almost_eq!(&tensor2 / &tensor1, &one / &correct);

            // inner strides: [non-unif], [0]
            let tensor1 = NdArray::<T>::randint([n, 3], T::one(), five);
            let tensor1 = tensor1.slice_along(Axis(-1), 0..2);

            let correct: Vec<T> = tensor1.flatiter().map(|lhs| lhs / five).collect();
            let correct = NdArray::new(correct).reshape([n, 2]);

            assert_almost_eq!(&tensor1 / &tensor2, correct);
            assert_almost_eq!(&tensor1 / five, correct);
            assert_almost_eq!(&tensor2 / &tensor1, &one / &correct);
        }
    }
);

#[test]
fn test_broadcast_add() {
    let tensor1 = NdArray::new([[1, 2], [3, 4], [5, 6]]);
    let tensor2 = NdArray::new([2, 4]);

    let correct = NdArray::new([[3, 6], [5, 8], [7, 10]]);
    let output = tensor1 + tensor2;

    assert_eq!(output, correct);
}

#[test]
fn test_scalar_add1() {
    let tensor1 = NdArray::new([[1, 2], [3, 4], [5, 6]]);
    let correct = NdArray::new([[6, 7], [8, 9], [10, 11]]);
    assert_eq!(&tensor1 + 5, correct);
    assert_eq!(&tensor1 + NdArray::scalar(5), correct);
    assert_eq!(NdArray::scalar(5) + tensor1, correct);

    let tensor1 = NdArray::scalar(5);
    let correct = NdArray::scalar(10);
    assert_eq!(&tensor1 + 5, correct);
    assert_eq!(&tensor1 + NdArray::scalar(5), correct);
    assert_eq!(NdArray::scalar(5) + tensor1, correct);
}

#[test]
fn test_add_lifetimes() {
    let output;

    {
        let tensor1 = NdArray::new([[1, 2], [3, 4], [5, 6]]);
        let tensor2 = NdArray::new([2, 4]);
        output = tensor1 + tensor2;
    }

    let correct = NdArray::new([[3, 6], [5, 8], [7, 10]]);
    assert_eq!(output, correct);
}

#[test]
fn test_scalar_add_lifetimes() {
    let output;

    {
        let tensor1 = NdArray::new([[1, 2], [3, 4], [5, 6]]);
        output = tensor1 + 5;
    }

    let correct = NdArray::new([[6, 7], [8, 9], [10, 11]]);
    assert_eq!(output, correct);
}

#[test]
fn test_add_references() {
    let tensor1 = NdArray::new([[1, 2], [3, 4], [5, 6]]);

    let correct = NdArray::new([[6, 7], [8, 9], [10, 11]]);
    let output = &tensor1 + 5;

    assert_eq!(tensor1, tensor1);
    assert_eq!(output, correct);
}

#[test]
fn test_remainder() {
    let tensor1 = NdArray::new([[1, 3], [2, 2], [3, 7]]);
    let tensor2 = NdArray::new([2, 4]);

    let correct = NdArray::new([[1, 3], [0, 2], [1, 3]]);
    let output = tensor1 % tensor2;

    assert_eq!(output, correct);
}

#[test]
fn test_bitand() {
    let tensor1 = NdArray::new([[1, 3], [2, 2], [3, 7]]);
    let tensor2 = NdArray::new([1, 2]);

    let correct = NdArray::new([[1, 2], [0, 2], [1, 2]]);
    let output = tensor1 & tensor2;

    assert_eq!(output, correct);
}

#[test]
fn test_bitor() {
    let tensor1 = NdArray::new([[1, 3], [2, 2], [3, 7]]);
    let tensor2 = NdArray::new([1, 2]);

    let correct = NdArray::new([[1, 3], [3, 2], [3, 7]]);
    let output = tensor1 | tensor2;

    assert_eq!(output, correct);
}

#[test]
fn test_shl() {
    let tensor1 = NdArray::new([[1, 2], [4, 8], [16, 32]]);
    let tensor2 = NdArray::new([1, 2]);

    let correct = NdArray::new([[2, 8], [8, 32], [32, 128]]);
    let output = tensor1 << tensor2;

    assert_eq!(output, correct);
}

#[test]
fn test_shr() {
    let tensor1 = NdArray::new([[2, 4], [8, 16], [32, 64]]);
    let tensor2 = NdArray::new([1, 2]);

    let correct = NdArray::new([[1, 1], [4, 4], [16, 16]]);
    let output = tensor1 >> tensor2;

    assert_eq!(output, correct);
}

#[test]
fn test_iadd() {
    let mut tensor1 = NdArray::new([[1, 2], [3, 4], [5, 6]]);
    let tensor2 = NdArray::new([2, 4]);

    let correct = NdArray::new([[3, 6], [5, 8], [7, 10]]);
    tensor1 += &tensor2;
    assert_eq!(tensor1, correct);

    let correct = NdArray::new([[5, 10], [7, 12], [9, 14]]);
    tensor1 += tensor2;
    assert_eq!(tensor1, correct);

    let correct = NdArray::new([[10, 15], [12, 17], [14, 19]]);
    tensor1 += 5;
    assert_eq!(tensor1, correct);
}

#[test]
fn test_isub() {
    let mut tensor1 = NdArray::new([[1, 2], [3, 4], [5, 6]]);
    let tensor2 = NdArray::new([2, 4]);

    let correct = NdArray::new([[-1, -2], [1, 0], [3, 2]]);
    tensor1 -= &tensor2;
    assert_eq!(tensor1, correct);

    let correct = NdArray::new([[-3, -6], [-1, -4], [1, -2]]);
    tensor1 -= tensor2;
    assert_eq!(tensor1, correct);

    let correct = NdArray::new([[-4, -7], [-2, -5], [0, -3]]);
    tensor1 -= 1;
    assert_eq!(tensor1, correct);
}

#[test]
#[should_panic]
fn test_broadcast_panic() {
    let tensor1 = NdArray::new([[1, 2], [3, 4], [5, 6]]);
    let tensor2 = NdArray::new([2, 4, 6]);
    let _ = tensor1 + tensor2;
}

#[test]
#[should_panic]
fn test_iadd_panic() {
    let mut tensor1 = NdArray::new([2, 4, 6]);
    let tensor2 = NdArray::new([[1, 2], [3, 4], [5, 6]]);
    tensor1 += &tensor2;
}

#[test]
#[should_panic]
fn test_viewonly_panic() {
    let tensor1 = NdArray::new([2, 4, 6]);
    let mut tensor1 = tensor1.broadcast_to(&[3, 3]);

    let tensor2 = NdArray::new([[1, 2, 3], [3, 4, 5], [5, 6, 7]]);
    tensor1 += &tensor2;
}
