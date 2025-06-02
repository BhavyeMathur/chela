use chela::*;
use num::{NumCast, Zero};
use paste::paste;

test_for_all_numeric_dtypes!(
    test_fill, {
        let mut a = NdArray::<T>::zeros([3, 5, 3]);

        assert!(a.flatiter().all(|x| x == T::zero()));
        a.fill(NumCast::from(5).unwrap());
        assert!(a.flatiter().all(|x| x == NumCast::from(5).unwrap()));

        for n in 1..23 {
            let mut a = NdArray::<T>::zeros([n]);

            assert!(a.flatiter().all(|x| x == T::zero()));
            a.fill(NumCast::from(5).unwrap());
            assert!(a.flatiter().all(|x| x == NumCast::from(5).unwrap()));
        }
    }
);


test_for_all_numeric_dtypes!(
    test_fill_slice, {
        // uniform stride but non-contiguous
        let a = NdArray::<T>::zeros([3, 5]);
        a.slice(s![.., 1]).fill(NumCast::from(5).unwrap());
        a.slice(s![1.., 0]).fill(NumCast::from(2).unwrap());

        let correct = NdArray::new([[0, 5, 0, 0, 0], [2, 5, 0, 0, 0], [2, 5, 0, 0, 0]]).astype::<T>();
        assert_eq!(a, correct);

        // non-uniform stride and non-contiguous
        let a = NdArray::<T>::zeros([3, 5]);
        a.slice(s![.., 1..4]).fill(NumCast::from(5).unwrap());

        let correct = NdArray::new([[0, 5, 5, 5, 0], [0, 5, 5, 5, 0], [0, 5, 5, 5, 0]]).astype::<T>();
        assert_eq!(a, correct);
    }
);

test_for_signed_int_dtypes!(
    test_fill_signed_slice, {
        let a = NdArray::<T>::zeros([3, 5]);
        a.slice(s![.., 1]).fill(NumCast::from(-5).unwrap());
        a.slice(s![1.., 0]).fill(NumCast::from(2).unwrap());
        a.slice(s![..2, 3..]).fill(NumCast::from(-7).unwrap());

        let correct = NdArray::new([[0, -5, 0, -7, -7], [2, -5, 0, -7, -7], [2, -5, 0, 0, 0]]).astype::<T>();
        assert_eq!(a, correct);
    }
);

#[test]
fn test_fill_slice_bool() {
    let a: NdArray<bool> = NdArray::zeros([3, 5]);
    a.slice(s![1, ..]).fill(true);

    let correct: NdArray<bool> = NdArray::new([
        [false, false, false, false, false],
        [true, true, true, true, true],
        [false, false, false, false, false]
    ]);
    assert_eq!(a, correct);
}
