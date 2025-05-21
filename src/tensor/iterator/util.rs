use crate::traits::haslength::HasLength;

// returns a pair of vectors: one indexed by indices and the other with the remaining elements
pub(super) fn split_by_indices<T, I>(data: &[T], indices: I) -> (Vec<T>, Vec<T>)
where
    T: Copy,
    I: HasLength + IntoIterator<Item=isize>,
{
    let max_index = data.len();
    let mut bitset = vec![false; max_index];

    let mut selected = Vec::with_capacity(indices.len());
    let mut remaining = Vec::with_capacity(max_index - indices.len());

    for index in indices {
        if bitset[index as usize] {
            panic!("duplicate index encountered");
        }
        
        bitset[index as usize] = true;
    }

    for (&value, &bit) in data.iter().zip(bitset.iter()) {
        if bit {
            selected.push(value);
        } else {
            remaining.push(value);
        }
    }

    (selected, remaining)
}
