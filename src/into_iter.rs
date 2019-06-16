#[cfg(not(feature = "const_generics"))]
use crate::Array;
use crate::SmallVec;
use core::ptr;

/// An iterator that consumes a `SmallVec` and yields its items by value.
///
/// Returned from [`SmallVec::into_iter`][1].
///
/// [1]: struct.SmallVec.html#method.into_iter
pub struct IntoIter<A: Array> {
    pub(crate) data: SmallVec<A>,
    pub(crate) current: usize,
    pub(crate) end: usize,
}

impl<A: Array> Drop for IntoIter<A> {
    fn drop(&mut self) {
        for _ in self {}
    }
}

impl<A: Array> Iterator for IntoIter<A> {
    type Item = A::Item;

    #[inline]
    fn next(&mut self) -> Option<A::Item> {
        if self.current == self.end {
            None
        } else {
            unsafe {
                let current = self.current as isize;
                self.current += 1;
                Some(ptr::read(self.data.as_ptr().offset(current)))
            }
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let size = self.end - self.current;
        (size, Some(size))
    }
}

impl<A: Array> DoubleEndedIterator for IntoIter<A> {
    #[inline]
    fn next_back(&mut self) -> Option<A::Item> {
        if self.current == self.end {
            None
        } else {
            unsafe {
                self.end -= 1;
                Some(ptr::read(self.data.as_ptr().offset(self.end as isize)))
            }
        }
    }
}

impl<A: Array> ExactSizeIterator for IntoIter<A> {}
