use {
    crate::{
        Allocator,
        DropDealloc,
        Global,
        RawSmallVec,
        SmallVec,
        TaggedLen
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
pub struct IntoIter<T, const N: usize, A: Allocator = Global> {
    // # Safety
    //
    // `end` decides whether the data lives on the heap or not
    //
    // The members from begin..end are initialized
    raw: RawSmallVec<T, N>,
    allocator: A,
    begin: usize,
    end: TaggedLen<T>
}

// SAFETY: IntoIter has unique ownership of its contents.  Sending (or sharing)
// an `IntoIter<T, N>` is equivalent to sending (or sharing) a `SmallVec<T, N>`.
unsafe impl<T: Send, const N: usize, A: Allocator + Send> Send for IntoIter<T, N, A> {}
unsafe impl<T: Sync, const N: usize, A: Allocator + Sync> Sync for IntoIter<T, N, A> {}

impl<T, const N: usize, A: Allocator> IntoIter<T, N, A> {
    #[inline]
    pub const fn as_slice(&self) -> &[T] {
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
    pub const fn as_mut_slice(&mut self) -> &mut [T] {
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

impl<T, const N: usize, A: Allocator> Iterator for IntoIter<T, N, A> {
    type Item = T;

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

impl<T, const N: usize, A: Allocator> DoubleEndedIterator for IntoIter<T, N, A> {
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

impl<T, const N: usize, A: Allocator> ExactSizeIterator for IntoIter<T, N, A> {}
impl<T, const N: usize, A: Allocator> core::iter::FusedIterator for IntoIter<T, N, A> {}

impl<T, const N: usize, A: Allocator> Drop for IntoIter<T, N, A> {
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
                    size_bytes: capacity * size_of::<T>(),
                    align: align_of::<T>(),
                    allocator: &self.allocator
                })
            } else {
                None
            };
            core::ptr::slice_from_raw_parts_mut(ptr.add(begin), end - begin).drop_in_place();
        }
    }
}

impl<T: Clone, const N: usize, A: Allocator + Clone> Clone for IntoIter<T, N, A> {
    #[inline]
    fn clone(&self) -> IntoIter<T, N, A> {
        let mut vec = SmallVec {
            length: TaggedLen::new(0, false),
            raw: RawSmallVec::new(),
            allocator: self.allocator.clone()
        };

        vec.extend(self.as_slice());

        vec.into_iter()
    }
}

impl<T, const N: usize, A: Allocator> IntoIterator for SmallVec<T, N, A> {
    type IntoIter = IntoIter<T, N, A>;
    type Item = T;

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

impl<T: Debug, const N: usize, A: Allocator> Debug for IntoIter<T, N, A> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_tuple("IntoIter").field(&self.as_slice()).finish()
    }
}
