use core::{ptr, slice};

/// An iterator that removes the items from a `SmallVec` and yields them by value.
///
/// Returned from [`SmallVec::drain`][1].
///
/// [1]: struct.SmallVec.html#method.drain
pub struct Drain<'a, T: 'a> {
    pub(crate) iter: slice::IterMut<'a, T>,
}

impl<'a, T: 'a> Iterator for Drain<'a, T> {
    type Item = T;

    #[inline]
    fn next(&mut self) -> Option<T> {
        self.iter
            .next()
            .map(|reference| unsafe { ptr::read(reference) })
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }
}

impl<'a, T: 'a> DoubleEndedIterator for Drain<'a, T> {
    #[inline]
    fn next_back(&mut self) -> Option<T> {
        self.iter
            .next_back()
            .map(|reference| unsafe { ptr::read(reference) })
    }
}

impl<'a, T> ExactSizeIterator for Drain<'a, T> {}

impl<'a, T: 'a> Drop for Drain<'a, T> {
    fn drop(&mut self) {
        // Destroy the remaining elements.
        for _ in self.by_ref() {}
    }
}
