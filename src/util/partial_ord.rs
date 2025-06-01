use crate::absolute::Absolute;

pub(crate) fn partial_min<T: PartialOrd>(a: T, b: T) -> T {
    if a.partial_cmp(&b) == Some(std::cmp::Ordering::Less) { a } else { b }
}

pub(crate) fn partial_max<T: PartialOrd>(a: T, b: T) -> T {
    if a.partial_cmp(&b) == Some(std::cmp::Ordering::Greater) { a } else { b }
}

pub(crate) fn partial_min_magnitude<T: PartialOrd + Absolute>(val: T, acc: T) -> T {
    let val = val.abs();
    if acc.partial_cmp(&val) == Some(std::cmp::Ordering::Less) { acc } else { val }
}

pub(crate) fn partial_max_magnitude<T: PartialOrd + Absolute>(val: T, acc: T) -> T {
    let val = val.abs();
    if acc.partial_cmp(&val) == Some(std::cmp::Ordering::Greater) { acc } else { val }
}
