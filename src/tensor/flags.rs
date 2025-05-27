use bitflags::bitflags;

bitflags! {
    #[repr(transparent)]
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct TensorFlags: u8 {
        const Owned = 0b00000001;
        const Contiguous = 0b00000010;
        const UniformStride = 0b00000100;
        const Writeable = 0b00001000;
        const UserCreated = 0b00010000;
        const RequiresGrad = 0b00100000;
    }
}
