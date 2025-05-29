use crate::dtype::RawDataType;
use crate::NdArray;

impl<T1, T2> PartialEq<NdArray<'_, T1>> for NdArray<'_, T2>
where
    T1: RawDataType,
    T2: RawDataType + From<T1>,
{
    fn eq(&self, other: &NdArray<T1>) -> bool {
        if self.shape != other.shape {
            return false;
        }
        self.flatiter().zip(other.flatiter()).all(|(a, b)| a == b.into())
    }
}
