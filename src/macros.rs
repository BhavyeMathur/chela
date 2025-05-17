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
macro_rules! test_for_all_numeric_types {
    ($name:ident, $body:tt) => {
        implement_test_for_dtypes!($name, $body,
            f32, f64,
            i8, i16, i32, i64, i128,
            u8, u16, u32, u64, u128
        );
    };
}
