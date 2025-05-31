use chela::*;

macro_rules! test_astype_from {
    ($src_ty:ty) => {
        paste::paste! {
            #[test]
            fn [<test_astype_from_1d_ $src_ty>]() {
                let a = NdArray::new([0 as $src_ty, 1 as $src_ty, 2 as $src_ty, 3 as $src_ty, 4 as $src_ty]);

                let expected = NdArray::new([0i8, 1, 2, 3, 4]);
                let result = a.astype::<i8>();
                assert_eq!(expected, result);

                let expected = NdArray::new([0i16, 1, 2, 3, 4]);
                let result = a.astype::<i16>();
                assert_eq!(expected, result);

                let expected = NdArray::new([0i32, 1, 2, 3, 4]);
                let result = a.astype::<i32>();
                assert_eq!(expected, result);

                let expected = NdArray::new([0i64, 1, 2, 3, 4]);
                let result = a.astype::<i64>();
                assert_eq!(expected, result);

                let expected = NdArray::new([0i128, 1, 2, 3, 4]);
                let result = a.astype::<i128>();
                assert_eq!(expected, result);

                let expected = NdArray::new([0u8, 1, 2, 3, 4]);
                let result = a.astype::<u8>();
                assert_eq!(expected, result);

                let expected = NdArray::new([0u16, 1, 2, 3, 4]);
                let result = a.astype::<u16>();
                assert_eq!(expected, result);

                let expected = NdArray::new([0u32, 1, 2, 3, 4]);
                let result = a.astype::<u32>();
                assert_eq!(expected, result);

                let expected = NdArray::new([0u64, 1, 2, 3, 4]);
                let result = a.astype::<u64>();
                assert_eq!(expected, result);

                let expected = NdArray::new([0u128, 1, 2, 3, 4]);
                let result = a.astype::<u128>();
                assert_eq!(expected, result);

                let expected = NdArray::new([0f32, 1.0, 2.0, 3.0, 4.0]);
                let result = a.astype::<f32>();
                assert_eq!(expected, result);

                let expected = NdArray::new([0f64, 1.0, 2.0, 3.0, 4.0]);
                let result = a.astype::<f64>();
                assert_eq!(expected, result);
            }

            #[test]
            fn [<test_astype_from_2d_ $src_ty>]() {
                let a = NdArray::new([[0 as $src_ty, 1 as $src_ty], [2 as $src_ty, 3 as $src_ty]]);

                let expected = NdArray::new([[0i8, 1], [2, 3]]);
                let result = a.astype::<i8>();
                assert_eq!(expected, result);

                let expected = NdArray::new([[0i16, 1], [2, 3]]);
                let result = a.astype::<i16>();
                assert_eq!(expected, result);

                let expected = NdArray::new([[0i32, 1], [2, 3]]);
                let result = a.astype::<i32>();
                assert_eq!(expected, result);

                let expected = NdArray::new([[0i64, 1], [2, 3]]);
                let result = a.astype::<i64>();
                assert_eq!(expected, result);

                let expected = NdArray::new([[0i128, 1], [2, 3]]);
                let result = a.astype::<i128>();
                assert_eq!(expected, result);

                let expected = NdArray::new([[0u8, 1], [2, 3]]);
                let result = a.astype::<u8>();
                assert_eq!(expected, result);

                let expected = NdArray::new([[0u16, 1], [2, 3]]);
                let result = a.astype::<u16>();
                assert_eq!(expected, result);

                let expected = NdArray::new([[0u32, 1], [2, 3]]);
                let result = a.astype::<u32>();
                assert_eq!(expected, result);

                let expected = NdArray::new([[0u64, 1], [2, 3]]);
                let result = a.astype::<u64>();
                assert_eq!(expected, result);

                let expected = NdArray::new([[0u128, 1], [2, 3]]);
                let result = a.astype::<u128>();
                assert_eq!(expected, result);

                let expected = NdArray::new([[0f32, 1.0], [2.0, 3.0]]);
                let result = a.astype::<f32>();
                assert_eq!(expected, result);

                let expected = NdArray::new([[0f64, 1.0], [2.0, 3.0]]);
                let result = a.astype::<f64>();
                assert_eq!(expected, result);
            }
        }
    };
}

macro_rules! generate_astype_tests {
    ($($ty:ty),*) => {
        $( test_astype_from!($ty); )*
    };
}


generate_astype_tests!(
    i8, i16, i32, i64, i128,
    u8, u16, u32, u64, u128,
    f32, f64
);
