use bitflags::bitflags;

bitflags! {
    #[repr(transparent)]
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct TensorFlags: u8 {
        const Owned = 0b00000001;
        const Contiguous = 0b00000010;
    }
}
