use crate::recursive_trait_base_cases;

pub(crate) trait RawDataType {}

impl RawDataType for u8 {}
impl RawDataType for u16 {}
impl RawDataType for u32 {}
impl RawDataType for u64 {}
impl RawDataType for u128 {}

impl RawDataType for i8 {}
impl RawDataType for i16 {}
impl RawDataType for i32 {}
impl RawDataType for i64 {}
impl RawDataType for i128 {}

impl RawDataType for f32 {}
impl RawDataType for f64 {}

impl RawDataType for bool {}

pub(crate) trait RawData: Sized {
    type DType;
}

impl<A> RawData for Vec<A>
where
    A: RawDataType,
{
    type DType = A;
}

impl<A, const N: usize> RawData for [A; N]
where
    A: RawData,
{
    type DType = A;
}

macro_rules! raw_data_array_trait {
    ( $dtype:ty ) => {
        impl<const N: usize> RawData for [$dtype; N] {
            type DType = $dtype;
        }
    };
}

recursive_trait_base_cases!(raw_data_array_trait);
