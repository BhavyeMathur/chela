use crate::dtype::RawDataType;
use crate::{NdArray, NdArrayMethods};
use std::fmt;


// TODO this is a temporary implementation
impl<T: RawDataType> fmt::Debug for NdArray<'_, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fn fmt_inner<T: fmt::Debug>(f: &mut fmt::Formatter<'_>,
                                    shape: &[usize],
                                    iter: &mut dyn Iterator<Item=T>,
                                    level: usize) -> fmt::Result {
            if shape.is_empty() {
                return Ok(write!(f, "{:?}", iter.next().unwrap())?);
            }

            if shape.len() == 1 {
                write!(f, "[")?;
                for i in 0..shape[0] {
                    if i > 0 { write!(f, ", ")?; }
                    write!(f, "{:?}", iter.next().unwrap())?
                }
                write!(f, "]")
            } else {
                write!(f, "[")?;
                for i in 0..shape[0] {
                    if i > 0 {
                        write!(f, ",{:indent$}", "", indent = (level + 1) * 2)?;
                    } else {
                        write!(f, "{:indent$}", "", indent = (level + 1) * 2)?;
                    }
                    fmt_inner(f, &shape[1..], iter, level + 1)?;
                }
                write!(f, "{:indent$}]", "", indent = level * 2)
            }
        }

        fmt_inner(f, self.shape(), &mut self.flatiter(), 0)
    }
}
