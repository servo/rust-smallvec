use core::marker::PhantomData;

/// Vec guarantees that its length is always less than [`isize::MAX`] in
/// *bytes*.
///
/// For a non-ZST, this means that the length is less than `isize::MAX` objects,
/// which implies we have at least one free bit we can use. We use the least
/// significant bit for the tag. And store the length in the `usize::BITS - 1`
/// most significant bits.
///
/// For a ZST, we never use the heap, so we just store the length directly.
#[repr(transparent)]
pub struct TaggedLen<T>(usize, PhantomData<T>);

// We don't use `#[derive(Clone, Copy)]` instead because `T` doesn't need to be
// `Copy` or `Clone`.
impl<T> Clone for TaggedLen<T> {
    #[inline]
    fn clone(&self) -> Self {
        *self
    }
}

impl<T> Copy for TaggedLen<T> {}

#[allow(clippy::len_without_is_empty)]
impl<T> TaggedLen<T> {
    const MAX_LEN: usize = usize::MAX >> Self::SHIFT;
    const SHIFT: u32 = (size_of::<T>() != 0) as u32;
    const TAG: usize = Self::SHIFT as usize;

    #[inline(always)]
    pub const fn new(len: usize, on_heap: bool) -> Self {
        debug_assert!(len < Self::MAX_LEN);
        debug_assert!(!on_heap || Self::TAG != 0);
        Self(
            (len << Self::SHIFT) | ((on_heap as usize) & Self::TAG),
            PhantomData
        )
    }

    #[inline(always)]
    pub const fn len(self) -> usize {
        self.0 >> Self::SHIFT
    }

    #[inline(always)]
    #[must_use]
    pub const fn on_heap(self) -> bool {
        self.0 & Self::TAG != 0
    }

    #[inline(always)]
    pub const fn parts(self) -> (usize, bool) {
        (self.0 >> Self::SHIFT, (self.0 & Self::TAG) != 0)
    }

    /// # Safety
    ///
    /// current len+n must be smaller than MAX_LEN
    #[inline(always)]
    pub const unsafe fn add(&mut self, n: usize) {
        debug_assert!(self.len() + n < Self::MAX_LEN);
        self.0 += n << Self::SHIFT;
    }

    /// # Safety
    ///
    /// current len must be greater equal than n
    #[inline(always)]
    pub const unsafe fn sub(&mut self, n: usize) {
        debug_assert!(self.len() >= n);
        self.0 -= n << Self::SHIFT;
    }
}
