// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use crate::{Allocator, SmallVec};

/// An iterator which uses a closure to determine if an element should be
/// removed.
///
/// Returned from [`SmallVec::extract_if`][1].
///
/// [1]: struct.SmallVec.html#method.extract_if
pub struct ExtractIf<'a, T, const N: usize, A: Allocator, F>
where F: FnMut(&mut T) -> bool
{
    pub(crate) vec: &'a mut SmallVec<T, N, A>,
    /// The index of the item that will be inspected by the next call to `next`.
    pub(crate) idx: usize,
    /// Elements at and beyond this point will be retained. Must be equal or
    /// smaller than `old_len`.
    pub(crate) end: usize,
    /// The number of items that have been drained (removed) thus far.
    pub(crate) del: usize,
    /// The original length of `vec` prior to draining.
    pub(crate) old_len: usize,
    /// The filter test predicate.
    pub(crate) pred: F
}

impl<T, const N: usize, A: Allocator, F> core::fmt::Debug for ExtractIf<'_, T, N, A, F>
where
    F: FnMut(&mut T) -> bool,
    T: core::fmt::Debug
{
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_tuple("ExtractIf")
            .field(&self.vec.as_slice())
            .finish()
    }
}
impl<T, F, const N: usize, A: Allocator> Iterator for ExtractIf<'_, T, N, A, F>
where F: FnMut(&mut T) -> bool
{
    type Item = T;

    fn next(&mut self) -> Option<T> {
        unsafe {
            while self.idx < self.end {
                let i = self.idx;
                // SAFETY: `i < self.end <= self.old_len`
                let cur = self.vec.as_mut_ptr().add(i);
                let drained = (self.pred)(&mut *cur);
                // Update the index *after* the predicate is called. If the
                // index is updated prior and the predicate
                // panics, the element at this index would be
                // leaked.
                self.idx += 1;
                if drained {
                    self.del += 1;
                    return Some(core::ptr::read(cur));
                } else if self.del > 0 {
                    // SAFETY: `self.del <= i` therefore `i - self.del` is valid
                    core::ptr::copy_nonoverlapping(cur, cur.sub(self.del), 1);
                }
            }
            None
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (0, Some(self.end - self.idx))
    }
}

impl<T, F, const N: usize, A: Allocator> Drop for ExtractIf<'_, T, N, A, F>
where F: FnMut(&mut T) -> bool
{
    fn drop(&mut self) {
        unsafe {
            if self.idx < self.old_len && self.del > 0 {
                // This is a pretty messed up state, and there isn't really an
                // obviously right thing to do. We don't want to keep trying
                // to execute `pred`, so we just backshift all the unprocessed
                // elements and tell the vec that they still exist. The
                // backshift is required to prevent a
                // double-drop of the last successfully
                // drained item prior to a panic in the predicate.
                let ptr = self.vec.as_mut_ptr();
                let src = ptr.add(self.idx);
                let dst = src.sub(self.del);
                let tail_len = self.old_len - self.idx;
                src.copy_to(dst, tail_len);
            }
            self.vec.set_len(self.old_len - self.del);
        }
    }
}
