pub trait Absolute: Copy {
    fn abs(&self) -> Self {
        *self
    }
}

impl Absolute for u8 {}
impl Absolute for u16 {}
impl Absolute for u32 {}
impl Absolute for u64 {}
impl Absolute for u128 {}
impl Absolute for usize {}

impl Absolute for i8 {
    fn abs(&self) -> Self {
        num::Signed::abs(self)
    }
}

impl Absolute for i16 {
    fn abs(&self) -> Self {
        num::Signed::abs(self)
    }
}

impl Absolute for i32 {
    fn abs(&self) -> Self {
        num::Signed::abs(self)
    }
}

impl Absolute for i64 {
    fn abs(&self) -> Self {
        num::Signed::abs(self)
    }
}

impl Absolute for i128 {
    fn abs(&self) -> Self {
        num::Signed::abs(self)
    }
}

impl Absolute for isize {
    fn abs(&self) -> Self {
        num::Signed::abs(self)
    }
}

impl Absolute for f32 {
    fn abs(&self) -> Self {
        num::Signed::abs(self)
    }
}

impl Absolute for f64 {
    fn abs(&self) -> Self {
        num::Signed::abs(self)
    }
}
