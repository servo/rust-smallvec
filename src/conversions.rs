use crate::SmallVec;
use alloc::vec::Vec;
use core::{mem::ManuallyDrop, ptr::copy_nonoverlapping};

impl<T: Clone, const N: usize> From<&[T]> for SmallVec<T, N> {
    #[inline]
    fn from(slice: &[T]) -> Self {
        if slice.len() > Self::inline_size() {
            // Standard Rust vectors are already specialized.
            Self::from_vec(Vec::from(slice))
        } else {
            // SAFETY: The precondition is checked in the initial comparison
            // above.
            unsafe {
                #[cfg(feature = "specialization")]
                {
                    <Self as crate::spec_traits::SpecFromSlice<T>>::spec_from(slice)
                }

                #[cfg(not(feature = "specialization"))]
                {
                    Self::from_slice_fallback(slice)
                }
            }
        }
    }
}

impl<T: Clone, const N: usize> From<&mut [T]> for SmallVec<T, N> {
    #[inline]
    fn from(slice: &mut [T]) -> Self {
        Self::from(slice as &[T])
    }
}

impl<T: Clone, const M: usize, const N: usize> From<&[T; M]> for SmallVec<T, N> {
    #[inline]
    fn from(slice: &[T; M]) -> Self {
        Self::from(slice as &[T])
    }
}

impl<T: Clone, const M: usize, const N: usize> From<&mut [T; M]> for SmallVec<T, N> {
    #[inline]
    fn from(slice: &mut [T; M]) -> Self {
        Self::from(slice as &[T])
    }
}

impl<T, const N: usize, const M: usize> From<[T; M]> for SmallVec<T, N> {
    fn from(array: [T; M]) -> Self {
        if M > N {
            // If M > N, we'd have to heap allocate anyway,
            // so delegate for Vec for the allocation.
            Self::from(Vec::from(array))
        } else {
            // M <= N
            let mut this = Self::new();
            debug_assert!(M <= this.capacity());
            let array = ManuallyDrop::new(array);
            // SAFETY: M <= this.capacity()
            unsafe {
                copy_nonoverlapping(array.as_ptr(), this.as_mut_ptr(), M);
                this.set_len(M);
            }
            this
        }
    }
}

impl<T, const N: usize, const M: usize> TryFrom<SmallVec<T, N>> for [T; M] {
    type Error = SmallVec<T, N>;

    #[inline]
    fn try_from(mut this: SmallVec<T, N>) -> Result<[T; M], SmallVec<T, N>> {
        if this.len() != M {
            Err(this)
        } else {
            // SAFETY: we release ownership of the elements we hold
            unsafe {
                this.set_len(0);
            }
            let ptr = this.as_ptr() as *const [T; M];
            // SAFETY: these elements are initialized since the length was `M`
            unsafe { Ok(ptr.read()) }
        }
    }
}

impl<T, const N: usize> From<Vec<T>> for SmallVec<T, N> {
    fn from(array: Vec<T>) -> Self {
        Self::from_vec(array)
    }
}
