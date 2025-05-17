use chela::*;

macro_rules! test_astype_from {
    ($src_ty:ty, $name:ident) => {
        #[test]
        fn $name() {
            let a = Tensor::from([0 as $src_ty, 1 as $src_ty, 2 as $src_ty, 3 as $src_ty, 4 as $src_ty]);

            let expected = Tensor::from([0i8, 1, 2, 3, 4]);
            let result = a.astype::<i8>();
            assert_eq!(expected, result);

            let expected = Tensor::from([0i16, 1, 2, 3, 4]);
            let result = a.astype::<i16>();
            assert_eq!(expected, result);

            let expected = Tensor::from([0i32, 1, 2, 3, 4]);
            let result = a.astype::<i32>();
            assert_eq!(expected, result);

            let expected = Tensor::from([0i64, 1, 2, 3, 4]);
            let result = a.astype::<i64>();
            assert_eq!(expected, result);

            let expected = Tensor::from([0i128, 1, 2, 3, 4]);
            let result = a.astype::<i128>();
            assert_eq!(expected, result);

            let expected = Tensor::from([0u8, 1, 2, 3, 4]);
            let result = a.astype::<u8>();
            assert_eq!(expected, result);

            let expected = Tensor::from([0u16, 1, 2, 3, 4]);
            let result = a.astype::<u16>();
            assert_eq!(expected, result);

            let expected = Tensor::from([0u32, 1, 2, 3, 4]);
            let result = a.astype::<u32>();
            assert_eq!(expected, result);

            let expected = Tensor::from([0u64, 1, 2, 3, 4]);
            let result = a.astype::<u64>();
            assert_eq!(expected, result);

            let expected = Tensor::from([0u128, 1, 2, 3, 4]);
            let result = a.astype::<u128>();
            assert_eq!(expected, result);

            let expected = Tensor::from([0f32, 1.0, 2.0, 3.0, 4.0]);
            let result = a.astype::<f32>();
            assert_eq!(expected, result);

            let expected = Tensor::from([0f64, 1.0, 2.0, 3.0, 4.0]);
            let result = a.astype::<f64>();
            assert_eq!(expected, result);
        }
    };
}

test_astype_from!(i8, test_astype_from_i8);
test_astype_from!(i16, test_astype_from_i16);
test_astype_from!(i32, test_astype_from_i32);
test_astype_from!(i64, test_astype_from_i64);
test_astype_from!(i128, test_astype_from_i128);

test_astype_from!(u8, test_astype_from_u8);
test_astype_from!(u16, test_astype_from_u16);
test_astype_from!(u32, test_astype_from_u32);
test_astype_from!(u64, test_astype_from_u64);
test_astype_from!(u128, test_astype_from_u128);

test_astype_from!(f32, test_astype_from_f32);
test_astype_from!(f64, test_astype_from_f64);
