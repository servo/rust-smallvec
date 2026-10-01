use {
    crate::{
        Allocator,
        Box,
        Global,
        SmallVec,
        Vec
    },
    core::{
        mem::ManuallyDrop,
        ptr::copy_nonoverlapping
    }
};

impl<T: Clone, const N: usize> From<&[T]> for SmallVec<T, N, Global> {
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
                    <Self as crate::specialization::SpecFromSlice<T>>::spec_from(slice)
                }

                #[cfg(not(feature = "specialization"))]
                {
                    Self::from_slice_fallback(slice)
                }
            }
        }
    }
}

impl<T: Clone, const N: usize> From<&mut [T]> for SmallVec<T, N, Global> {
    #[inline]
    fn from(slice: &mut [T]) -> Self {
        Self::from(slice as &[T])
    }
}

impl<T: Clone, const M: usize, const N: usize> From<&[T; M]> for SmallVec<T, N, Global> {
    #[inline]
    fn from(slice: &[T; M]) -> Self {
        Self::from(slice as &[T])
    }
}

impl<T: Clone, const M: usize, const N: usize> From<&mut [T; M]> for SmallVec<T, N, Global> {
    #[inline]
    fn from(slice: &mut [T; M]) -> Self {
        Self::from(slice as &[T])
    }
}

impl<T, const N: usize, const M: usize> From<[T; M]> for SmallVec<T, N, Global> {
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

impl<T, const N: usize, const M: usize, A: Allocator> TryFrom<SmallVec<T, N, A>> for [T; M] {
    type Error = SmallVec<T, N, A>;

    #[inline]
    fn try_from(mut this: SmallVec<T, N, A>) -> Result<[T; M], SmallVec<T, N, A>> {
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

impl<T, const N: usize> From<Vec<T>> for SmallVec<T, N, Global> {
    fn from(array: Vec<T>) -> Self {
        Self::from_vec(array)
    }
}

impl<T, const N: usize, A: Allocator> From<SmallVec<T, N, A>> for Vec<T> {
    fn from(this: SmallVec<T, N, A>) -> Self {
        let (length, on_heap) = this.length.parts();
        if !on_heap {
            let mut vec = Vec::with_capacity(length);
            let this = ManuallyDrop::new(this);
            // SAFETY: we create a new vector with sufficient capacity, copy our
            // elements into it to transfer ownership and then set
            // the length we don't drop the elements we previously
            // held
            unsafe {
                copy_nonoverlapping(this.raw.as_ptr_inline(), vec.as_mut_ptr(), length);
                vec.set_len(length);
            }
            vec
        } else {
            let this = ManuallyDrop::new(this);
            // SAFETY:
            // - `ptr` was created with the SmallVec's allocator
            // - `ptr` was created with the appropriate alignment for `T`
            // - the allocation pointed to by ptr is exactly cap * sizeof(T)
            // - `length` is less than or equal to `cap`
            // - the first `length` entries are proper `T`-values
            // - the allocation is not larger than `isize::MAX`
            unsafe {
                let (ptr, cap) = this.raw.heap;
                Vec::from_raw_parts(ptr.as_ptr(), length, cap)
            }
        }
    }
}

impl<T, const N: usize, A: Allocator> From<SmallVec<T, N, A>> for Box<[T]> {
    fn from(this: SmallVec<T, N, A>) -> Self {
        Vec::from(this).into_boxed_slice()
    }
}
