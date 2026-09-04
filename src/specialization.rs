use {
    crate::{
        DropGuard,
        IntoIter,
        SmallVec
    },
    core::ptr::copy_nonoverlapping
};

/// A trait for specializing the implementation of [`from_elem`].
///
/// [`from_elem`]: crate::from_elem
pub trait SpecFromElem<T> {
    /// Creates a `Smallvec` value where `elem` is repeated `n` times.
    /// This will use the inline storage, not the heap.
    ///
    /// # Safety
    ///
    /// The caller must ensure that `n <= Self::inline_size()`.
    unsafe fn spec_from_elem(elem: T, n: usize) -> Self;
}

impl<T: Clone, const N: usize> SpecFromElem<T> for SmallVec<T, N> {
    #[inline]
    default unsafe fn spec_from_elem(elem: T, n: usize) -> Self {
        // SAFETY: Safety conditions are identical.
        unsafe { SmallVec::from_elem_fallback(elem, n) }
    }
}

impl<T: Copy, const N: usize> SpecFromElem<T> for SmallVec<T, N> {
    unsafe fn spec_from_elem(elem: T, n: usize) -> Self {
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
pub trait SpecExtend<T, I> {
    fn spec_extend(&mut self, iter: I);
}

impl<T, I, const N: usize> SpecExtend<T, I> for SmallVec<T, N>
where I: Iterator<Item = T>
{
    #[inline]
    default fn spec_extend(&mut self, iter: I) {
        self.extend_fallback(iter);
    }
}

impl<T, I, const N: usize> SpecExtend<T, I> for SmallVec<T, N>
where I: core::iter::TrustedLen<Item = T>
{
    fn spec_extend(&mut self, iter: I) {
        let (_, Some(additional)) = iter.size_hint() else {
            panic!("capacity overflow")
        };
        self.reserve(additional);

        // SAFETY: A `TrustedLen` iterator provides accurate information
        // about its size, which was used to reserve additional memory.
        // This ensures that the access operations inside the loop always
        // operate on valid memory.
        unsafe {
            let len = self.len();
            let ptr = self.as_mut_ptr().add(len);
            let mut guard = DropGuard {
                ptr,
                len: 0
            };

            for x in iter {
                ptr.add(guard.len).write(x);
                guard.len += 1;
            }

            // The elements have been initialized in the loop above.
            self.set_len(len + guard.len);
            core::mem::forget(guard);
        }
    }
}

impl<T, const N: usize, const M: usize> SpecExtend<T, IntoIter<T, M>> for SmallVec<T, N> {
    fn spec_extend(&mut self, mut iter: IntoIter<T, M>) {
        let slice = iter.as_slice();
        let len = slice.len();
        let old_len = self.len();

        self.reserve(len);

        // SAFETY: Additional memory has been reserved above.
        // Therefore, the copy operates on valid memory.
        unsafe {
            let dst = self.as_mut_ptr().add(old_len);
            let src = slice.as_ptr();
            copy_nonoverlapping(src, dst, len);
        }

        // SAFETY: The elements were initialized above.
        unsafe {
            self.set_len(old_len + len);
        }

        // Mark the iterator as fully consumed.
        iter.begin = iter.end.len();
    }
}

impl<'a, T: 'a, const N: usize, I> SpecExtend<&'a T, I> for SmallVec<T, N>
where
    I: Iterator<Item = &'a T>,
    T: Clone
{
    #[inline]
    default fn spec_extend(&mut self, iterator: I) {
        self.spec_extend(iterator.cloned())
    }
}

impl<'a, T: 'a, const N: usize> SpecExtend<&'a T, core::slice::Iter<'a, T>> for SmallVec<T, N>
where T: Copy
{
    fn spec_extend(&mut self, iter: core::slice::Iter<'a, T>) {
        let slice = iter.as_slice();
        let len = slice.len();
        let old_len = self.len();

        self.reserve(len);

        // SAFETY: Additional memory has been reserved above.
        // Therefore, the copy operates on valid memory.
        unsafe {
            let dst = self.as_mut_ptr().add(old_len);
            let src = slice.as_ptr();
            copy_nonoverlapping(src, dst, len);
        }

        // SAFETY: The elements were initialized above.
        unsafe {
            self.set_len(old_len + len);
        }
    }
}

/// A trait for specializing the implementation of [`extend_from_within`].
///
/// [`extend_from_within`]: crate::SmallVec::extend_from_within
pub trait SpecExtendFromWithin<T> {
    /// Main worker for [`extend_from_within`].
    ///
    /// # Safety
    ///
    /// * The length of the vector is larger than or equal to `src.len()`.
    /// * The spare capacity of the vector is larger than or equal to
    ///   `src.len()`.
    ///
    /// [`extend_from_within`]: SmallVec::extend_from_within
    unsafe fn spec_extend_from_within(&mut self, src: core::ops::Range<usize>);
}

impl<T: Clone, const N: usize> SpecExtendFromWithin<T> for SmallVec<T, N> {
    default unsafe fn spec_extend_from_within(&mut self, src: core::ops::Range<usize>) {
        // SAFETY: Safety conditions are identical.
        unsafe {
            self.extend_from_within_fallback(src);
        }
    }
}

impl<T: Copy, const N: usize> SpecExtendFromWithin<T> for SmallVec<T, N> {
    unsafe fn spec_extend_from_within(&mut self, src: core::ops::Range<usize>) {
        let old_len = self.len();

        let start = src.start;
        let len = src.len();

        // SAFETY: The caller ensures that the vector has spare capacity
        // for at least `src.len()` elements. This is also the amount of
        // memory accessed when the data is copied.
        unsafe {
            let ptr = self.as_mut_ptr();
            let dst = ptr.add(old_len);
            let src = ptr.add(start);
            copy_nonoverlapping(src, dst, len);
        }

        // SAFETY: The elements were initialized above.
        unsafe {
            self.set_len(old_len + len);
        }
    }
}

/// A trait for specializing the implementation of [`FromIterator`].
///
/// [`clone_from`]: Clone::clone_from
pub trait SpecFromIterator<T, I> {
    fn spec_from_iter(iter: I) -> Self;
}

impl<T, I, const N: usize> SpecFromIterator<T, I> for SmallVec<T, N>
where I: Iterator<Item = T>
{
    #[inline]
    default fn spec_from_iter(iter: I) -> Self {
        Self::from_iter_fallback(iter)
    }
}

impl<T, I, const N: usize> SpecFromIterator<T, I> for SmallVec<T, N>
where I: core::iter::TrustedLen<Item = T>
{
    fn spec_from_iter(iter: I) -> Self {
        let mut v = match iter.size_hint() {
            (_, Some(upper)) => SmallVec::with_capacity(upper),
            // TrustedLen contract guarantees that `size_hint() == (_, None)` means that there
            // are more than `usize::MAX` elements.
            // Since the previous branch would eagerly panic if the capacity is too large
            // (via `with_capacity`) we do the same here.
            _ => panic!("capacity overflow")
        };
        // Reuse the extend specialization for TrustedLen.
        v.spec_extend(iter);
        v
    }
}

/// A trait for specializing the implementation of [`clone_from`].
///
/// [`clone_from`]: Clone::clone_from
pub trait SpecCloneFrom<T> {
    fn spec_clone_from(&mut self, source: &[T]);
}

impl<T: Clone, const N: usize> SpecCloneFrom<T> for SmallVec<T, N> {
    #[inline]
    default fn spec_clone_from(&mut self, source: &[T]) {
        self.clone_from_fallback(source);
    }
}

impl<T: Copy, const N: usize> SpecCloneFrom<T> for SmallVec<T, N> {
    fn spec_clone_from(&mut self, source: &[T]) {
        self.clear();
        self.extend_from_slice(source);
    }
}

/// A trait for specializing the implementation of [`From`]
/// with the source type being slices.
pub trait SpecFromSlice<T> {
    /// Creates a `SmallVec` value based on the contents of `slice`.
    /// This will use the inline storage, not the heap.
    ///
    /// # Safety
    ///
    /// The caller must ensure that `slice.len() <= Self::inline_size()`.
    unsafe fn spec_from(slice: &[T]) -> Self;
}

impl<T: Clone, const N: usize> SpecFromSlice<T> for SmallVec<T, N> {
    default unsafe fn spec_from(slice: &[T]) -> Self {
        // SAFETY: Safety conditions are identical.
        unsafe { Self::from_slice_fallback(slice) }
    }
}

impl<T: Copy, const N: usize> SpecFromSlice<T> for SmallVec<T, N> {
    unsafe fn spec_from(slice: &[T]) -> Self {
        let mut v = Self::new();

        let src = slice.as_ptr();
        let len = slice.len();
        let dst = v.as_mut_ptr();

        // SAFETY: The caller ensures that the slice length is smaller
        // than or equal to the inline length.
        unsafe {
            copy_nonoverlapping(src, dst, len);
        }

        // SAFETY: The elements were initialized above.
        unsafe {
            v.set_len(len);
        }

        v
    }
}
