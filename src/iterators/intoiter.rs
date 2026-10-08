use {
    crate::{
        Allocator,
        DropDealloc,
        Global,
        LocatedLength,
        RawSmallVec,
        SmallVec
    },
    core::{
        fmt::Debug,
        mem::{
            ManuallyDrop,
            align_of,
            size_of
        },
        ptr::NonNull
    }
};

/// An iterator that consumes a `SmallVec` and yields its items by value.
///
/// Returned from [`SmallVec::into_iter`][1].
///
/// [1]: struct.SmallVec.html#method.into_iter
pub struct IntoIter<Item, const INLINE: usize, Heap: Allocator = Global> {
    // # Safety
    //
    // `end` decides whether the data lives on the heap or not
    //
    // The members from begin..end are initialized
    raw: RawSmallVec<Item, INLINE>,
    allocator: Heap,
    begin: usize,
    end: LocatedLength<Item>
}

// SAFETY: IntoIter has unique ownership of its contents.  Sending (or sharing)
// an `IntoIter<Item, INLINE>` is equivalent to sending (or sharing) a
// `SmallVec<Item, INLINE, Global>`.
unsafe impl<Item: Send, const INLINE: usize, Heap: Allocator + Send> Send
    for IntoIter<Item, INLINE, Heap>
{
}
unsafe impl<Item: Sync, const INLINE: usize, Heap: Allocator + Sync> Sync
    for IntoIter<Item, INLINE, Heap>
{
}

impl<Item, const INLINE: usize, Heap: Allocator> IntoIter<Item, INLINE, Heap> {
    #[inline]
    pub const fn as_slice(&self) -> &[Item] {
        let (end, on_heap) = self.end.parts();
        // SAFETY: `end` tells which buffer is active, and the members in
        // `self.begin..end` are all initialized. So the pointer arithmetic is
        // valid, and so is the construction of the slice
        unsafe {
            let ptr = self.raw.as_ptr(on_heap);
            core::slice::from_raw_parts(ptr.add(self.begin), end - self.begin)
        }
    }

    #[inline]
    pub const fn as_mut_slice(&mut self) -> &mut [Item] {
        let (end, on_heap) = self.end.parts();
        // SAFETY: see above
        unsafe {
            let ptr = self.raw.as_mut_ptr(on_heap);
            core::slice::from_raw_parts_mut(ptr.add(self.begin), end - self.begin)
        }
    }

    #[cfg(feature = "specialization")]
    pub(crate) fn mark_consumed(&mut self) {
        self.begin = self.end.len();
    }
}

impl<Item, const INLINE: usize, Heap: Allocator> Iterator for IntoIter<Item, INLINE, Heap> {
    type Item = Item;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        let (end, on_heap) = self.end.parts();
        if self.begin == end {
            None
        } else {
            // SAFETY: see above
            unsafe {
                let ptr = self.raw.as_mut_ptr(on_heap);
                let value = ptr.add(self.begin).read();
                self.begin += 1;
                Some(value)
            }
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let size = self.end.len() - self.begin;
        (size, Some(size))
    }
}

impl<Item, const INLINE: usize, Heap: Allocator> DoubleEndedIterator
    for IntoIter<Item, INLINE, Heap>
{
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        let (end, on_heap) = self.end.parts();
        if self.begin == end {
            None
        } else {
            // SAFETY: see above
            unsafe {
                let ptr = self.raw.as_mut_ptr(on_heap);
                self.end.sub(1);
                let value = ptr.add(end - 1).read();
                Some(value)
            }
        }
    }
}

impl<Item, const INLINE: usize, Heap: Allocator> ExactSizeIterator
    for IntoIter<Item, INLINE, Heap>
{
}
impl<Item, const INLINE: usize, Heap: Allocator> core::iter::FusedIterator
    for IntoIter<Item, INLINE, Heap>
{
}

impl<Item, const INLINE: usize, Heap: Allocator> Drop for IntoIter<Item, INLINE, Heap> {
    fn drop(&mut self) {
        // SAFETY: see above
        unsafe {
            let (end, on_heap) = self.end.parts();
            let begin = self.begin;
            let ptr = self.raw.as_mut_ptr(on_heap);
            let _drop_dealloc = if on_heap {
                let capacity = self.raw.heap.1;
                Some(DropDealloc {
                    ptr: NonNull::new_unchecked(ptr as *mut u8),
                    size_bytes: capacity * size_of::<Item>(),
                    align: align_of::<Item>(),
                    allocator: &self.allocator
                })
            } else {
                None
            };
            core::ptr::slice_from_raw_parts_mut(ptr.add(begin), end - begin).drop_in_place();
        }
    }
}

impl<Item: Clone, const INLINE: usize, Heap: Allocator + Clone> Clone
    for IntoIter<Item, INLINE, Heap>
{
    #[inline]
    fn clone(&self) -> IntoIter<Item, INLINE, Heap> {
        let mut vec = SmallVec {
            length: LocatedLength::new(0, false),
            raw: RawSmallVec::new(),
            allocator: self.allocator.clone()
        };

        vec.extend(self.as_slice());

        vec.into_iter()
    }
}

impl<Item, const INLINE: usize, Heap: Allocator> IntoIterator for SmallVec<Item, INLINE, Heap> {
    type IntoIter = IntoIter<Item, INLINE, Heap>;
    type Item = Item;

    fn into_iter(self) -> Self::IntoIter {
        // SAFETY: we move out of this.raw by reading the value at its address,
        // which is fine since we don't drop it
        unsafe {
            // Set SmallVec length to zero as `IntoIter` drop handles dropping
            // of the elements
            let this = ManuallyDrop::new(self);
            IntoIter {
                raw: (&raw const this.raw).read(),
                allocator: (&raw const this.allocator).read(),
                begin: 0,
                end: this.length
            }
        }
    }
}

impl<Item: Debug, const INLINE: usize, Heap: Allocator> Debug for IntoIter<Item, INLINE, Heap> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_tuple("IntoIter").field(&self.as_slice()).finish()
    }
}
