use chela::*;
use num::{NumCast, Zero};
use paste::paste;

#[test]
fn test() {
    type T = f32;
    let n = 2;

    let tensor = NdArray::arange(0, n).squeeze().astype::<T>();

    let correct: Vec<T> = tensor.flatiter().map(|lhs| -lhs).collect();
    let correct = NdArray::new(correct).squeeze();

    assert_eq!(-&tensor, correct);
    assert_eq!(-tensor, correct);
}


test_for_signed_dtypes!(
    test_neg, {
        // inner stride: [0]
        let tensor = NdArray::scalar(127).astype::<T>();
        let correct = NdArray::scalar(-127).astype::<T>();
        assert_eq!(-&tensor, correct);
        assert_eq!(-tensor, correct);

        // inner stride: [1]
        for n in 1..23 {
            let tensor = NdArray::arange(0, n).squeeze().astype::<T>();

            let correct: Vec<T> = tensor.flatiter().map(|lhs| -lhs).collect();
            let correct = NdArray::new(correct).squeeze();

            assert_eq!(-&tensor, correct);
            assert_eq!(-tensor, correct);
        }
    }
);

test_for_signed_dtypes!(
    test_neg_non_contiguous, {
        let high = NumCast::from(127).unwrap();

        for n in 1..23 {
            // inner stride: [k]
            let tensor = NdArray::<T>::randint([n, 2], T::zero(), high);
            let tensor = tensor.slice_along(Axis(-1), 0);

            let correct: Vec<T> = tensor.flatiter().map(|lhs| -lhs).collect();
            let correct = NdArray::new(correct);

            assert_eq!(-&tensor, correct);
            assert_eq!(-tensor, correct);

            // inner stride: [non-unif]
            let tensor = NdArray::<T>::randint([n, 3], T::zero(), high);
            let tensor = tensor.slice_along(Axis(-1), 0..2);

            let correct: Vec<T> = tensor.flatiter().map(|lhs| -lhs).collect();
            let correct = NdArray::new(correct).reshape([n, 2]);

            assert_eq!(-&tensor, correct);
            assert_eq!(-tensor, correct);
        }
    }
);
