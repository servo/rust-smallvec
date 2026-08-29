//! Iterator implementation extracted from `lib.rs`.

use super::*;

/// An iterator that consumes a `SmallVec` and yields its items by value.
///
/// Returned from [`SmallVec::into_iter`][1].
///
/// [1]: struct.SmallVec.html#method.into_iter
pub struct IntoIter<T, const N: usize> {
    // # Safety
    //
    // `end` decides whether the data lives on the heap or not
    //
    // The members from begin..end are initialized
    pub(crate) raw: RawSmallVec<T, N>,
    pub(crate) begin: usize,
    pub(crate) end: TaggedLen<T>,
    pub(crate) _marker: PhantomData<T>,
}

// SAFETY: IntoIter has unique ownership of its contents.  Sending (or sharing)
// an `IntoIter<T, N>` is equivalent to sending (or sharing) a `SmallVec<T, N>`.
unsafe impl<T, const N: usize> Send for IntoIter<T, N> where T: Send {}
unsafe impl<T, const N: usize> Sync for IntoIter<T, N> where T: Sync {}

impl<T, const N: usize> IntoIter<T, N> {
    #[inline]
    const fn as_ptr(&self) -> *const T {
        let on_heap = self.end.on_heap();
        if on_heap {
            // SAFETY: vector is on the heap
            unsafe { self.raw.as_ptr_heap() }
        } else {
            self.raw.as_ptr_inline()
        }
    }

    #[inline]
    const fn as_mut_ptr(&mut self) -> *mut T {
        let on_heap = self.end.on_heap();
        if on_heap {
            // SAFETY: vector is on the heap
            unsafe { self.raw.as_mut_ptr_heap() }
        } else {
            self.raw.as_mut_ptr_inline()
        }
    }

    #[inline]
    pub const fn as_slice(&self) -> &[T] {
        // SAFETY: The members in self.begin..self.end.value() are all initialized
        // So the pointer arithmetic is valid, and so is the construction of the slice
        unsafe {
            let ptr = self.as_ptr();
            core::slice::from_raw_parts(ptr.add(self.begin), self.end.value() - self.begin)
        }
    }

    #[inline]
    pub const fn as_mut_slice(&mut self) -> &mut [T] {
        // SAFETY: see above
        unsafe {
            let ptr = self.as_mut_ptr();
            core::slice::from_raw_parts_mut(ptr.add(self.begin), self.end.value() - self.begin)
        }
    }
}

impl<T, const N: usize> Iterator for IntoIter<T, N> {
    type Item = T;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.begin == self.end.value() {
            None
        } else {
            // SAFETY: see above
            unsafe {
                let ptr = self.as_mut_ptr();
                let value = ptr.add(self.begin).read();
                self.begin += 1;
                Some(value)
            }
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let size = self.end.value() - self.begin;
        (size, Some(size))
    }
}

impl<T, const N: usize> DoubleEndedIterator for IntoIter<T, N> {
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        let mut end = self.end.value();
        if self.begin == end {
            None
        } else {
            // SAFETY: see above
            unsafe {
                let ptr = self.as_mut_ptr();
                let on_heap = self.end.on_heap();
                end -= 1;
                self.end = TaggedLen::new(end, on_heap);
                let value = ptr.add(end).read();
                Some(value)
            }
        }
    }
}
impl<T, const N: usize> ExactSizeIterator for IntoIter<T, N> {}
impl<T, const N: usize> core::iter::FusedIterator for IntoIter<T, N> {}

impl<T, const N: usize> Drop for IntoIter<T, N> {
    fn drop(&mut self) {
        // SAFETY: see above
        unsafe {
            let on_heap = self.end.on_heap();
            let begin = self.begin;
            let end = self.end.value();
            let ptr = self.as_mut_ptr();
            let _drop_dealloc = if on_heap {
                let capacity = self.raw.heap.1;
                Some(DropDealloc {
                    ptr: NonNull::new_unchecked(ptr as *mut u8),
                    size_bytes: capacity * size_of::<T>(),
                    align: align_of::<T>(),
                })
            } else {
                None
            };
            core::ptr::slice_from_raw_parts_mut(ptr.add(begin), end - begin).drop_in_place();
        }
    }
}

impl<T: Clone, const N: usize> Clone for IntoIter<T, N> {
    #[inline]
    fn clone(&self) -> IntoIter<T, N> {
        SmallVec::from(self.as_slice()).into_iter()
    }
}

impl<T: Debug, const N: usize> Debug for IntoIter<T, N> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_tuple("IntoIter").field(&self.as_slice()).finish()
    }
}
