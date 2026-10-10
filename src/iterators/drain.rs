use crate::{Allocator, Global, SmallVec, SmallVecError};

/// An iterator that removes the items from a `SmallVec` and yields them by
/// value.
///
/// Returned from [`SmallVec::drain`][1].
///
/// [1]: struct.SmallVec.html#method.drain
pub struct Drain<'valid, Item: 'valid, const INLINE: usize, Heap: Allocator = Global> {
    // `vec` points to a valid object within its lifetime.
    // This is ensured by the fact that we're holding an iterator to its items.
    //
    // # Safety
    //
    // Members in vec[tail_start..tail_start + tail_len] are initialized
    // even though vec has length < tail_start
    pub(crate) tail_start: usize,
    pub(crate) tail_len: usize,
    pub(crate) iter: core::slice::Iter<'valid, Item>,
    pub(crate) vec: core::ptr::NonNull<SmallVec<Item, INLINE, Heap>>,
}

impl<'valid, Item: 'valid, const INLINE: usize, Heap: Allocator> Iterator
    for Drain<'valid, Item, INLINE, Heap>
{
    type Item = Item;

    #[inline]
    fn next(&mut self) -> Option<Item> {
        // SAFETY: we shrunk the length of the vector so it no longer owns these
        // items, and we can take ownership of them.
        self.iter
            .next()
            .map(|reference| unsafe { core::ptr::read(reference) })
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }
}

impl<'valid, Item: 'valid, const INLINE: usize, Heap: Allocator> DoubleEndedIterator
    for Drain<'valid, Item, INLINE, Heap>
{
    #[inline]
    fn next_back(&mut self) -> Option<Item> {
        // SAFETY: see above
        self.iter
            .next_back()
            .map(|reference| unsafe { core::ptr::read(reference) })
    }
}

impl<Item, const INLINE: usize, Heap: Allocator> ExactSizeIterator
    for Drain<'_, Item, INLINE, Heap>
{
    #[inline]
    fn len(&self) -> usize {
        self.iter.len()
    }
}

impl<Item, const INLINE: usize, Heap: Allocator> core::iter::FusedIterator
    for Drain<'_, Item, INLINE, Heap>
{
}

impl<'valid, Item: 'valid, const INLINE: usize, Heap: Allocator> Drop
    for Drain<'valid, Item, INLINE, Heap>
{
    fn drop(&mut self) {
        /// Moves back the un-`Drain`ed elements to restore the original `Vec`.
        struct DropGuard<'guard, 'valid, Item, const INLINE: usize, Heap: Allocator>(
            &'guard mut Drain<'valid, Item, INLINE, Heap>,
        );

        impl<'guard, 'valid, Item, const INLINE: usize, Heap: Allocator> Drop
            for DropGuard<'guard, 'valid, Item, INLINE, Heap>
        {
            fn drop(&mut self) {
                if self.0.tail_len > 0 {
                    unsafe {
                        let source_vec = self.0.vec.as_mut();
                        // memmove back untouched tail, update to new length
                        let start = source_vec.len();
                        let tail = self.0.tail_start;
                        if tail != start {
                            let ptr = source_vec.as_mut_ptr();
                            let source = ptr.add(tail);
                            let destination = ptr.add(start);
                            core::ptr::copy(source, destination, self.0.tail_len);
                        }
                        source_vec.length.add(self.0.tail_len);
                    }
                }
            }
        }

        let iter = core::mem::take(&mut self.iter);
        let drop_len = iter.len();

        let mut vec = self.vec;

        if SmallVec::<Item, INLINE, Heap>::IS_ZST {
            // ZSTs have no identity, so we don't need to move them around, we
            // only need to drop the correct amount. this can be
            // achieved by manipulating the Vec length instead of
            // moving values out from `iter`.
            unsafe {
                let vec = vec.as_mut();
                let old_len = vec.len();
                vec.length.add(drop_len + self.tail_len);
                vec.truncate(old_len + self.tail_len);
            }

            return;
        }

        // ensure elements are moved back into their appropriate places, even
        // when drop_in_place panics
        let _guard = DropGuard(self);

        if drop_len == 0 {
            return;
        }

        // as_slice() must only be called when iter.len() is > 0 because
        // it also gets touched by vec::Splice which may turn it into a dangling
        // pointer which would make it and the vec pointer point to
        // different allocations which would lead to invalid pointer
        // arithmetic below.
        let drop_ptr = iter.as_slice().as_ptr();

        unsafe {
            // drop_ptr comes from a slice::Iter which only gives us a &[Item]
            // but for drop_in_place a pointer with mutable
            // provenance is necessary. Therefore we must
            // reconstruct it from the original vec but also avoid
            // creating a &mut to the front since that could
            // invalidate raw pointers to it which some unsafe code
            // might rely on.
            let vec_ptr = vec.as_mut().as_mut_ptr();
            // May be replaced with the line below later, once this crate's MSRV
            // is >= 1.87. let drop_offset =
            // drop_ptr.offset_from_unsigned(vec_ptr);
            let drop_offset = drop_ptr.offset_from(vec_ptr) as usize;
            let to_drop = core::ptr::slice_from_raw_parts_mut(vec_ptr.add(drop_offset), drop_len);
            core::ptr::drop_in_place(to_drop);
        }
    }
}

impl<Item, const INLINE: usize, Heap: Allocator> Drain<'_, Item, INLINE, Heap> {
    #[must_use]
    pub fn as_slice(&self) -> &[Item] {
        self.iter.as_slice()
    }

    /// The range from `self.vec.length` to `self.tail_start` contains elements
    /// that have been moved out.
    /// Fill that range as much as possible with new elements from the
    /// `replace_with` iterator. Returns `true` if we filled the entire
    /// range. (`replace_with.next()` didn’t return `None`.)
    pub(crate) unsafe fn fill<I: Iterator<Item = Item>>(&mut self, replace_with: &mut I) -> bool {
        let vec = unsafe { self.vec.as_mut() };
        let range_end = self.tail_start;

        while vec.len() < range_end {
            let Some(new_item) = replace_with.next() else {
                return false;
            };
            let length = vec.len();
            // SAFETY: length < tail_start <= capacity
            unsafe {
                vec.as_mut_ptr().add(length).write(new_item);
                vec.length.add(1);
            }
        }
        true
    }

    /// Makes room for inserting more elements before the tail.
    #[track_caller]
    pub(crate) unsafe fn move_tail(&mut self, additional: usize) {
        let vec = unsafe { self.vec.as_mut() };
        let length = self.tail_start + self.tail_len;

        // Include the tail when reserving so it survives a reallocation.
        let old_len = vec.len();
        unsafe { vec.set_len(length) }
        let result = vec.try_reserve(additional);
        // Restore the prefix length before a reservation error can panic.
        unsafe { vec.set_len(old_len) };
        result.unwrap_or_else(SmallVecError::handle);

        let new_tail_start = self.tail_start + additional;
        unsafe {
            let ptr = vec.as_mut_ptr();
            core::ptr::copy(
                ptr.add(self.tail_start),
                ptr.add(new_tail_start),
                self.tail_len,
            );
        }
        self.tail_start = new_tail_start;
    }
}

impl<Item: core::fmt::Debug, const INLINE: usize, Heap: Allocator> core::fmt::Debug
    for Drain<'_, Item, INLINE, Heap>
{
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_tuple("Drain").field(&self.iter.as_slice()).finish()
    }
}
