use {
    crate::{Allocator, DropGuard, Global, IntoIter, SmallVec},
    core::ptr::copy_nonoverlapping,
};

/// A trait for specializing the implementation of [`from_elem`].
///
/// [`from_elem`]: crate::from_elem
pub trait SpecFromElem<Item> {
    /// Creates a `Smallvec` value where `elem` is repeated `n` times.
    /// This will use the inline storage, not the heap.
    ///
    /// # Safety
    ///
    /// The caller must ensure that `n <= Self::inline_size()`.
    unsafe fn spec_from_elem(elem: Item, n: usize) -> Self;
}

impl<Item: Clone, const INLINE: usize> SpecFromElem<Item> for SmallVec<Item, INLINE, Global> {
    #[inline]
    default unsafe fn spec_from_elem(elem: Item, n: usize) -> Self {
        // SAFETY: Safety conditions are identical.
        unsafe { SmallVec::from_elem_fallback(elem, n) }
    }
}

impl<Item: Copy, const INLINE: usize> SpecFromElem<Item> for SmallVec<Item, INLINE, Global> {
    unsafe fn spec_from_elem(elem: Item, n: usize) -> Self {
        let mut result = Self::new();

        if n > 0 {
            let ptr = result.raw.as_mut_ptr_inline();

            // SAFETY: The caller ensures that the first `n`
            // is smaller than the inline size.
            unsafe {
                for i in 0..n {
                    ptr.add(i).write(elem);
                }
            }
        }

        // SAFETY: The first `n` elements of the vector
        // have been initialized in the loop above.
        unsafe {
            result.set_len(n);
        }

        result
    }
}

/// A trait for specializing the implementations of [`Extend`] and
/// [`extend_from_slice`].
///
/// [`extend_from_slice`]: crate::SmallVec::extend_from_slice
pub trait SpecExtend<Item, I> {
    fn spec_extend(&mut self, iter: I);
}

impl<Item, I, const INLINE: usize, Heap: Allocator> SpecExtend<Item, I>
    for SmallVec<Item, INLINE, Heap>
where
    I: Iterator<Item = Item>,
{
    #[inline]
    default fn spec_extend(&mut self, iter: I) {
        self.extend_fallback(iter);
    }
}

impl<Item, I, const INLINE: usize, Heap: Allocator> SpecExtend<Item, I>
    for SmallVec<Item, INLINE, Heap>
where
    I: core::iter::TrustedLen<Item = Item>,
{
    fn spec_extend(&mut self, iter: I) {
        let (_, Some(additional)) = iter.size_hint() else {
            panic!("capacity overflow")
        };
        self.reserve(additional);

        // SAFETY: Heap `TrustedLen` iterator provides accurate information
        // about its size, which was used to reserve additional memory.
        // This ensures that the access operations inside the loop always
        // operate on valid memory.
        unsafe {
            let length = self.len();
            let ptr = self.as_mut_ptr().add(length);
            let mut guard = DropGuard { ptr, length: 0 };

            for x in iter {
                ptr.add(guard.length).write(x);
                guard.length += 1;
            }

            // The elements have been initialized in the loop above.
            self.set_len(length + guard.length);
            core::mem::forget(guard);
        }
    }
}

impl<Item, const INLINE: usize, const M: usize, Heap: Allocator>
    SpecExtend<Item, IntoIter<Item, M, Heap>> for SmallVec<Item, INLINE, Heap>
{
    fn spec_extend(&mut self, mut iter: IntoIter<Item, M, Heap>) {
        let slice = iter.as_slice();
        let length = slice.len();
        let old_len = self.len();

        self.reserve(length);

        // SAFETY: Additional memory has been reserved above.
        // Therefore, the copy operates on valid memory.
        unsafe {
            let destination = self.as_mut_ptr().add(old_len);
            let source = slice.as_ptr();
            copy_nonoverlapping(source, destination, length);
        }

        // SAFETY: The elements were initialized above.
        unsafe {
            self.set_len(old_len + length);
        }

        // Mark the iterator as fully consumed.
        iter.mark_consumed();
    }
}

impl<'valid, Item: 'valid, const INLINE: usize, I, Heap: Allocator> SpecExtend<&'valid Item, I>
    for SmallVec<Item, INLINE, Heap>
where
    I: Iterator<Item = &'valid Item>,
    Item: Clone,
{
    #[inline]
    default fn spec_extend(&mut self, iterator: I) {
        self.spec_extend(iterator.cloned())
    }
}

impl<'valid, Item: 'valid, const INLINE: usize, Heap: Allocator>
    SpecExtend<&'valid Item, core::slice::Iter<'valid, Item>> for SmallVec<Item, INLINE, Heap>
where
    Item: Copy,
{
    fn spec_extend(&mut self, iter: core::slice::Iter<'valid, Item>) {
        let slice = iter.as_slice();
        let length = slice.len();
        let old_len = self.len();

        self.reserve(length);

        // SAFETY: Additional memory has been reserved above.
        // Therefore, the copy operates on valid memory.
        unsafe {
            let destination = self.as_mut_ptr().add(old_len);
            let source = slice.as_ptr();
            copy_nonoverlapping(source, destination, length);
        }

        // SAFETY: The elements were initialized above.
        unsafe {
            self.set_len(old_len + length);
        }
    }
}

/// A trait for specializing the implementation of [`extend_from_within`].
///
/// [`extend_from_within`]: crate::SmallVec::extend_from_within
pub trait SpecExtendFromWithin<Item> {
    /// Main worker for [`extend_from_within`].
    ///
    /// # Safety
    ///
    /// * The length of the vector is larger than or equal to `source.len()`.
    /// * The spare capacity of the vector is larger than or equal to
    ///   `source.len()`.
    ///
    /// [`extend_from_within`]: SmallVec::extend_from_within
    unsafe fn spec_extend_from_within(&mut self, source: core::ops::Range<usize>);
}

impl<Item: Clone, const INLINE: usize, Heap: Allocator> SpecExtendFromWithin<Item>
    for SmallVec<Item, INLINE, Heap>
{
    default unsafe fn spec_extend_from_within(&mut self, source: core::ops::Range<usize>) {
        // SAFETY: Safety conditions are identical.
        unsafe {
            self.extend_from_within_fallback(source);
        }
    }
}

impl<Item: Copy, const INLINE: usize, Heap: Allocator> SpecExtendFromWithin<Item>
    for SmallVec<Item, INLINE, Heap>
{
    unsafe fn spec_extend_from_within(&mut self, source: core::ops::Range<usize>) {
        let old_len = self.len();

        let start = source.start;
        let length = source.len();

        // SAFETY: The caller ensures that the vector has spare capacity
        // for at least `source.len()` elements. This is also the amount of
        // memory accessed when the data is copied.
        unsafe {
            let ptr = self.as_mut_ptr();
            let destination = ptr.add(old_len);
            let source = ptr.add(start);
            copy_nonoverlapping(source, destination, length);
        }

        // SAFETY: The elements were initialized above.
        unsafe {
            self.set_len(old_len + length);
        }
    }
}

/// A trait for specializing the implementation of [`FromIterator`].
///
/// [`clone_from`]: Clone::clone_from
pub trait SpecFromIterator<Item, I> {
    fn spec_from_iter(iter: I) -> Self;
}

impl<Item, I, const INLINE: usize> SpecFromIterator<Item, I> for SmallVec<Item, INLINE, Global>
where
    I: Iterator<Item = Item>,
{
    #[inline]
    default fn spec_from_iter(iter: I) -> Self {
        Self::from_iter_fallback(iter)
    }
}

impl<Item, I, const INLINE: usize> SpecFromIterator<Item, I> for SmallVec<Item, INLINE, Global>
where
    I: core::iter::TrustedLen<Item = Item>,
{
    fn spec_from_iter(iter: I) -> Self {
        let mut v = match iter.size_hint() {
            (_, Some(upper)) => SmallVec::with_capacity(upper),
            // TrustedLen contract guarantees that `size_hint() == (_, None)` means that there
            // are more than `usize::MAX` elements.
            // Since the previous branch would eagerly panic if the capacity is too large
            // (via `with_capacity`) we do the same here.
            _ => panic!("capacity overflow"),
        };
        // Reuse the extend specialization for TrustedLen.
        v.spec_extend(iter);
        v
    }
}

/// A trait for specializing the implementation of [`clone_from`].
///
/// [`clone_from`]: Clone::clone_from
pub trait SpecCloneFrom<Item> {
    fn spec_clone_from(&mut self, source: &[Item]);
}

impl<Item: Clone, const INLINE: usize, Heap: Allocator> SpecCloneFrom<Item>
    for SmallVec<Item, INLINE, Heap>
{
    #[inline]
    default fn spec_clone_from(&mut self, source: &[Item]) {
        self.clone_from_fallback(source);
    }
}

impl<Item: Copy, const INLINE: usize, Heap: Allocator> SpecCloneFrom<Item>
    for SmallVec<Item, INLINE, Heap>
{
    fn spec_clone_from(&mut self, source: &[Item]) {
        self.clear();
        self.extend_from_slice(source);
    }
}

/// A trait for specializing the implementation of [`From`]
/// with the source type being slices.
pub trait SpecFromSlice<Item> {
    /// Creates a `SmallVec` value based on the contents of `slice`.
    /// This will use the inline storage, not the heap.
    ///
    /// # Safety
    ///
    /// The caller must ensure that `slice.len() <= Self::inline_size()`.
    unsafe fn spec_from(slice: &[Item]) -> Self;
}

impl<Item: Clone, const INLINE: usize> SpecFromSlice<Item> for SmallVec<Item, INLINE, Global> {
    default unsafe fn spec_from(slice: &[Item]) -> Self {
        // SAFETY: Safety conditions are identical.
        unsafe { Self::from_slice_fallback(slice) }
    }
}

impl<Item: Copy, const INLINE: usize> SpecFromSlice<Item> for SmallVec<Item, INLINE, Global> {
    unsafe fn spec_from(slice: &[Item]) -> Self {
        let mut v = Self::new();

        let source = slice.as_ptr();
        let length = slice.len();
        let destination = v.as_mut_ptr();

        // SAFETY: The caller ensures that the slice length is smaller
        // than or equal to the inline length.
        unsafe {
            copy_nonoverlapping(source, destination, length);
        }

        // SAFETY: The elements were initialized above.
        unsafe {
            v.set_len(length);
        }

        v
    }
}
