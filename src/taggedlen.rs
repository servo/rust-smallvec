use core::marker::PhantomData;

/// Vec guarantees that its length is always less than [`isize::MAX`] in
/// *bytes*.
///
/// For a non ZST, this means that the length is less than `isize::MAX` objects,
/// which implies we have at least one free bit we can use. We use the least
/// significant bit for the tag. And store the length in the `usize::BITS - 1`
/// most significant bits.
///
/// For a ZST, we never use the heap, so we just store the length directly.
#[repr(transparent)]
pub struct TaggedLen<T>(usize, PhantomData<T>);

// Clone and Copy must be manually implemented because the generic interferes
// with the derive attribute implementations.
impl<T> Clone for TaggedLen<T> {
    fn clone(&self) -> Self { Self(self.0, PhantomData) }

    fn clone_from(&mut self, source: &Self) { self.0 = source.0; }
}

impl<T> Copy for TaggedLen<T> {}

impl<T> TaggedLen<T> {
    pub const fn new(len: usize, on_heap: bool) -> Self {
        if size_of::<T>() == 0 {
            debug_assert!(!on_heap);
            Self(len, PhantomData)
        } else {
            debug_assert!(len < isize::MAX as usize);
            Self((len << 1) | on_heap as usize, PhantomData)
        }
    }

    #[must_use]
    pub const fn on_heap(self) -> bool { return (size_of::<T>() != 0) && ((self.0 & 1usize) == 1); }

    pub const fn value(self) -> usize {
        return if size_of::<T>() == 0 {
            self.0
        } else {
            self.0 >> 1
        };
    }
}
