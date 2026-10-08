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

impl<Item: Clone, const INLINE: usize> From<&[Item]> for SmallVec<Item, INLINE, Global> {
    #[inline]
    fn from(slice: &[Item]) -> Self {
        if slice.len() > Self::inline_size() {
            // Standard Rust vectors are already specialized.
            Self::from_vec(Vec::from(slice))
        } else {
            // SAFETY: The precondition is checked in the initial comparison
            // above.
            unsafe {
                #[cfg(feature = "specialization")]
                {
                    <Self as crate::specialization::SpecFromSlice<Item>>::spec_from(slice)
                }

                #[cfg(not(feature = "specialization"))]
                {
                    Self::from_slice_fallback(slice)
                }
            }
        }
    }
}

impl<Item: Clone, const INLINE: usize> From<&mut [Item]> for SmallVec<Item, INLINE, Global> {
    #[inline]
    fn from(slice: &mut [Item]) -> Self {
        Self::from(slice as &[Item])
    }
}

impl<Item: Clone, const M: usize, const INLINE: usize> From<&[Item; M]>
    for SmallVec<Item, INLINE, Global>
{
    #[inline]
    fn from(slice: &[Item; M]) -> Self {
        Self::from(slice as &[Item])
    }
}

impl<Item: Clone, const M: usize, const INLINE: usize> From<&mut [Item; M]>
    for SmallVec<Item, INLINE, Global>
{
    #[inline]
    fn from(slice: &mut [Item; M]) -> Self {
        Self::from(slice as &[Item])
    }
}

impl<Item, const INLINE: usize, const M: usize> From<[Item; M]> for SmallVec<Item, INLINE, Global> {
    fn from(array: [Item; M]) -> Self {
        if M > INLINE {
            // If M > INLINE, we'd have to heap allocate anyway,
            // so delegate for Vec for the allocation.
            Self::from(Vec::from(array))
        } else {
            // M <= INLINE
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

impl<Item, const INLINE: usize, const M: usize, Heap: Allocator>
    TryFrom<SmallVec<Item, INLINE, Heap>> for [Item; M]
{
    type Error = SmallVec<Item, INLINE, Heap>;

    #[inline]
    fn try_from(
        mut this: SmallVec<Item, INLINE, Heap>
    ) -> Result<[Item; M], SmallVec<Item, INLINE, Heap>> {
        if this.len() != M {
            Err(this)
        } else {
            // SAFETY: we release ownership of the elements we hold
            unsafe {
                this.set_len(0);
            }
            let ptr = this.as_ptr() as *const [Item; M];
            // SAFETY: these elements are initialized since the length was `M`
            unsafe { Ok(ptr.read()) }
        }
    }
}

impl<Item, const INLINE: usize> From<Vec<Item>> for SmallVec<Item, INLINE, Global> {
    fn from(array: Vec<Item>) -> Self {
        Self::from_vec(array)
    }
}

impl<Item, const INLINE: usize, Heap: Allocator> From<SmallVec<Item, INLINE, Heap>> for Vec<Item> {
    fn from(this: SmallVec<Item, INLINE, Heap>) -> Self {
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
            // - `ptr` was created with the appropriate alignment for `Item`
            // - the allocation pointed to by ptr is exactly cap * sizeof(Item)
            // - `length` is less than or equal to `cap`
            // - the first `length` entries are proper `Item`-values
            // - the allocation is not larger than `isize::MAX`
            unsafe {
                let (ptr, cap) = this.raw.heap;
                Vec::from_raw_parts(ptr.as_ptr(), length, cap)
            }
        }
    }
}

impl<Item, const INLINE: usize, Heap: Allocator> From<SmallVec<Item, INLINE, Heap>>
    for Box<[Item]>
{
    fn from(this: SmallVec<Item, INLINE, Heap>) -> Self {
        Vec::from(this).into_boxed_slice()
    }
}

impl<T, const N: usize> From<Box<[T]>> for SmallVec<T, N> {
    fn from(boxed: Box<[T]>) -> Self {
        Self::from_vec(boxed.into_vec())
    }
}
