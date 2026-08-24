<<<<<<< HEAD
use {
    super::SmallVec,
    bytes::{buf::UninitSlice, BufMut},
};
<<<<<<< HEAD
=======
use bytes::{buf::UninitSlice, BufMut};
use super::SmallVec;
>>>>>>> 7fa0992 (feat: added bytes, serde, std, taggedlen files)

=======
>>>>>>> 239c751 (refactor: conversions file)
unsafe impl<const N: usize> BufMut for SmallVec<u8, N> {
    #[inline]
    fn remaining_mut(&self) -> usize {
        // A vector can never have more than isize::MAX bytes
        isize::MAX as usize - self.len()
    }
<<<<<<< HEAD
<<<<<<< HEAD
=======

>>>>>>> 7fa0992 (feat: added bytes, serde, std, taggedlen files)
=======
>>>>>>> 239c751 (refactor: conversions file)
    #[inline]
    unsafe fn advance_mut(&mut self, cnt: usize) {
        let len = self.len();
        let remaining = self.capacity() - len;
<<<<<<< HEAD
<<<<<<< HEAD
        if remaining < cnt {
            panic!("advance out of bounds: the len is {remaining} but advancing by {cnt}");
        }
        // Addition will not overflow since the sum is at most the capacity.
        self.set_len(len + cnt);
    }
=======

=======
>>>>>>> 239c751 (refactor: conversions file)
        if remaining < cnt {
            panic!("advance out of bounds: the len is {remaining} but advancing by {cnt}");
        }
        // Addition will not overflow since the sum is at most the capacity.
        self.set_len(len + cnt);
    }
<<<<<<< HEAD

>>>>>>> 7fa0992 (feat: added bytes, serde, std, taggedlen files)
=======
>>>>>>> 239c751 (refactor: conversions file)
    #[inline]
    fn chunk_mut(&mut self) -> &mut UninitSlice {
        if self.capacity() == self.len() {
            self.reserve(64); // Grow the smallvec
        }
<<<<<<< HEAD
<<<<<<< HEAD
        let cap = self.capacity();
        let len = self.len();
=======

        let cap = self.capacity();
        let len = self.len();

>>>>>>> 7fa0992 (feat: added bytes, serde, std, taggedlen files)
=======
        let cap = self.capacity();
        let len = self.len();
>>>>>>> 239c751 (refactor: conversions file)
        let ptr = self.as_mut_ptr();
        // SAFETY: Since `ptr` is valid for `cap` bytes, `ptr.add(len)` must be
        // valid for `cap - len` bytes. The subtraction will not underflow since
        // `len <= cap`.
        unsafe { UninitSlice::from_raw_parts_mut(ptr.add(len), cap - len) }
    }
<<<<<<< HEAD
<<<<<<< HEAD
=======

>>>>>>> 7fa0992 (feat: added bytes, serde, std, taggedlen files)
=======
>>>>>>> 239c751 (refactor: conversions file)
    // Specialize these methods so they can skip checking `remaining_mut`
    // and `advance_mut`.
    #[inline]
    fn put<T: bytes::Buf>(&mut self, mut src: T)
    where
        Self: Sized,
    {
        // In case the src isn't contiguous, reserve upfront.
        self.reserve(src.remaining());
<<<<<<< HEAD
<<<<<<< HEAD
=======

>>>>>>> 7fa0992 (feat: added bytes, serde, std, taggedlen files)
=======
>>>>>>> 239c751 (refactor: conversions file)
        while src.has_remaining() {
            let s = src.chunk();
            let l = s.len();
            self.extend_from_slice(s);
            src.advance(l);
        }
    }
<<<<<<< HEAD
<<<<<<< HEAD
=======

>>>>>>> 7fa0992 (feat: added bytes, serde, std, taggedlen files)
=======
>>>>>>> 239c751 (refactor: conversions file)
    #[inline]
    fn put_slice(&mut self, src: &[u8]) {
        self.extend_from_slice(src);
    }
<<<<<<< HEAD
<<<<<<< HEAD
=======

>>>>>>> 7fa0992 (feat: added bytes, serde, std, taggedlen files)
=======
>>>>>>> 239c751 (refactor: conversions file)
    #[inline]
    fn put_bytes(&mut self, val: u8, cnt: usize) {
        // If the addition overflows, then the `resize` will fail.
        let new_len = self.len().saturating_add(cnt);
        self.resize(new_len, val);
    }
}
