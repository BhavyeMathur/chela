pub trait RawDataType: Clone + Copy + PartialEq {}

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