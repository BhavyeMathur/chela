// TODO make sure this doesn't export the macro to the public API

#[macro_export(local_inner_macros)]
macro_rules! recursive_trait_base_cases {
    ( $macro_name: ident ) => {
        $macro_name!(i8);
        $macro_name!(i16);
        $macro_name!(i32);
        $macro_name!(i64);
        $macro_name!(i128);

        $macro_name!(u8);
        $macro_name!(u16);
        $macro_name!(u32);
        $macro_name!(u64);
        $macro_name!(u128);

        $macro_name!(f32);
        $macro_name!(f64);

        $macro_name!(bool);
    };
}
