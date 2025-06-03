// TODO make sure this doesn't export the macro to the public API

#[macro_export(local_inner_macros)]
macro_rules! recursive_trait_base_cases {
    ( $macro_name: ident ) => {
        $macro_name!(i8);
        $macro_name!(i16);
        $macro_name!(i32);
        $macro_name!(i64);
        $macro_name!(i128);
        $macro_name!(isize);

        $macro_name!(u8);
        $macro_name!(u16);
        $macro_name!(u32);
        $macro_name!(u64);
        $macro_name!(u128);
        $macro_name!(usize);

        $macro_name!(f32);
        $macro_name!(f64);

        $macro_name!(bool);
    };
}

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
macro_rules! test_for_float_dtypes {
    ($name:ident, $body:tt) => {
        implement_test_for_dtypes!($name, $body,
            f32, f64
        );
    };
}

#[macro_export]
macro_rules! test_for_common_signed_int_dtypes {
    ($name:ident, $body:tt) => {
        implement_test_for_dtypes!($name, $body,
            i32, i64, i128, isize
        );
    };
}

#[macro_export]
macro_rules! test_for_common_unsigned_int_dtypes {
    ($name:ident, $body:tt) => {
        implement_test_for_dtypes!($name, $body,
            u32, u64, u128
        );
    };
}

#[macro_export]
macro_rules! test_for_signed_int_dtypes {
    ($name:ident, $body:tt) => {
        test_for_common_signed_int_dtypes!($name, $body);
        implement_test_for_dtypes!($name, $body,
            i8, i16
        );
    };
}

#[macro_export]
macro_rules! test_for_unsigned_int_dtypes {
    ($name:ident, $body:tt) => {
        test_for_common_unsigned_int_dtypes!($name, $body);
        implement_test_for_dtypes!($name, $body,
            u8, u16
        );
    };
}

#[macro_export]
macro_rules! test_for_signed_dtypes {
    ($name:ident, $body:tt) => {
        test_for_float_dtypes!($name, $body);
        test_for_signed_int_dtypes!($name, $body);
    };
}

#[macro_export]
macro_rules! test_for_unsigned_dtypes {
    ($name:ident, $body:tt) => {
        test_for_unsigned_int_dtypes!($name, $body);
    };
}

#[macro_export]
macro_rules! test_for_integer_dtypes {
    ($name:ident, $body:tt) => {
        test_for_signed_int_dtypes!($name, $body);
        test_for_unsigned_int_dtypes!($name, $body);
    };
}

#[macro_export]
macro_rules! test_for_common_integer_dtypes {
    ($name:ident, $body:tt) => {
        test_for_common_signed_int_dtypes!($name, $body);
        test_for_common_unsigned_int_dtypes!($name, $body);
    };
}

#[macro_export]
macro_rules! test_for_all_numeric_dtypes {
    ($name:ident, $body:tt) => {
        test_for_integer_dtypes!($name, $body);
        test_for_float_dtypes!($name, $body);
    };
}

#[macro_export]
macro_rules! test_for_common_numeric_dtypes {
    ($name:ident, $body:tt) => {
        test_for_common_integer_dtypes!($name, $body);
        test_for_float_dtypes!($name, $body);
    };
}


#[macro_export]
macro_rules! test_for_all_dtypes {
    ($name:ident, $body:tt) => {
        test_for_all_numeric_dtypes!($name, $body);
        implement_test_for_dtypes!($name, $body,
            bool
        );
    };
}

#[macro_export]
macro_rules! assert_almost_eq {
    ($left:expr, $right:expr, $tol:expr) => {
        if ((&($left) - &($right)).max().flatiter().next().unwrap() as f64).abs() > $tol {
            assert_eq!($left, $right);
        }
    };

    ($left:expr, $right:expr) => {
        assert_almost_eq!($left, $right, 1e-5);
    };
}

#[macro_export]
macro_rules! first_n_elements {
    ($arr:expr, $n:expr) => {{
        &$arr[0..$n].try_into().unwrap()
    }};
}

#[macro_export]
macro_rules! impl_default_trait_for_dtypes {
    ($trait_name:ident, $($default_dtypes:ty),*) => {
        $(
            impl $trait_name for $default_dtypes {}
        )*
    };
}
