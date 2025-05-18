#[macro_export]
macro_rules! implement_test_for_dtypes {
    ($name:ident, $body:block, $($t:ty),*) => {
        $(
            paste! {
                #[test]
                fn [<$name _ $t>]() {
                    type T = $t;
                    $body
                }
            }
        )*
    };
}

#[macro_export]
macro_rules! test_for_all_float_types {
    ($name:ident, $body:tt) => {
        implement_test_for_dtypes!($name, $body,
            f32, f64
        );
    };
}

#[macro_export]
macro_rules! test_for_signed_integer_types {
    ($name:ident, $body:tt) => {
        implement_test_for_dtypes!($name, $body,
            i8, i16, i32, i64, i128
        );
    };
}

#[macro_export]
macro_rules! test_for_unsigned_integer_types {
    ($name:ident, $body:tt) => {
        implement_test_for_dtypes!($name, $body,
            u8, u16, u32, u64, u128
        );
    };
}

#[macro_export]
macro_rules! test_for_signed_types {
    ($name:ident, $body:tt) => {
        test_for_all_float_types!($name, $body);
        test_for_signed_integer_types!($name, $body);
    };
}

#[macro_export]
macro_rules! test_for_all_integer_types {
    ($name:ident, $body:tt) => {
        test_for_signed_integer_types!($name, $body);
        test_for_unsigned_integer_types!($name, $body);
    };
}

#[macro_export]
macro_rules! test_for_all_numeric_types {
    ($name:ident, $body:tt) => {
        test_for_all_integer_types!($name, $body);
        test_for_all_float_types!($name, $body);
    };
}

#[macro_export]
macro_rules! test_for_dtypes {
    ($name:ident, $body:tt) => {
        test_for_all_numeric_types!($name, $body);
        implement_test_for_dtypes!($name, $body,
            bool
        );
    };
}

#[macro_export]
macro_rules! assert_almost_eq {
    ($left:expr, $right:expr) => {
        assert!((($left) - ($right)).max() < 0.00001);
    };
}
