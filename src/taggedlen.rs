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
    pub const fn new(length: usize, on_heap: bool) -> Self {
        debug_assert!(length < Self::MAX_LEN);
        debug_assert!(!on_heap || Self::TAG != 0);
        Self(
            (length << Self::SHIFT) | ((on_heap as usize) & Self::TAG),
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

    #[inline(always)]
    pub const fn set_location<const ON: bool>(&mut self) {
        if Self::TAG != 0 {
            self.0 = (self.0 & !Self::TAG) | ON as usize;
        }
    }

    /// # Safety
    ///
    /// current length+n must be smaller than MAX_LEN on 64-bit target
    #[inline(always)]
    pub const unsafe fn add(&mut self, n: usize) {
        #[cfg(any(debug_assertions, not(target_pointer_width = "64")))]
        {
            #[cold]
            #[inline(never)]
            const fn assert_failed() {
                panic!("smallvec length overflow")
            }
            match self.len().checked_add(n) {
                Some(value) => {
                    if value > Self::MAX_LEN {
                        assert_failed()
                    }
                }
                None => assert_failed()
            }
        }
        self.0 += n << Self::SHIFT;
    }

    /// # Safety
    ///
    /// current length must be greater equal than n
    #[inline(always)]
    pub const unsafe fn sub(&mut self, n: usize) {
        debug_assert!(self.len() >= n);
        self.0 -= n << Self::SHIFT;
    }
}
