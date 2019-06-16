/// Trait to be implemented by a collection that can be extended from a slice
///
/// ## Example
///
/// ```rust
/// use smallvec::{ExtendFromSlice, SmallVec};
///
/// fn initialize<V: ExtendFromSlice<u8>>(v: &mut V) {
///     v.extend_from_slice(b"Test!");
/// }
///
/// let mut vec = Vec::new();
/// initialize(&mut vec);
/// assert_eq!(&vec, b"Test!");
///
/// let mut small_vec = SmallVec::<[u8; 8]>::new();
/// initialize(&mut small_vec);
/// assert_eq!(&small_vec as &[_], b"Test!");
/// ```
pub trait ExtendFromSlice<T> {
    /// Extends a collection from a slice of its element type
    fn extend_from_slice(&mut self, other: &[T]);
}

impl<T: Clone> ExtendFromSlice<T> for Vec<T> {
    fn extend_from_slice(&mut self, other: &[T]) {
        Vec::extend_from_slice(self, other)
    }
}
