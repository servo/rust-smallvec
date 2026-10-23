#[cfg(feature = "specialization")]
use super::specialization::SpecFromSlice;
use {
    super::{
        SmallVec,
        allocator::{
            Allocator,
            boxed::Box,
            global::Global,
            like::Like,
            vec::Vec
        },
        locatedlength::LocatedLength,
        rawsmallvec::RawSmallVec
    },
    core::{
        mem::ManuallyDrop,
        ptr::{
            NonNull,
            copy_nonoverlapping
        }
    }
};

impl<Item: Clone, const INLINE: usize> From<&[Item]> for SmallVec<Item, INLINE, Global> {
    #[inline]
    fn from(slice: &[Item]) -> Self {
        if slice.len() > Self::inline_size() {
            // Standard Rust vectors are already specialized.
            <Vec<Item, Global> as Like>::Type::from(slice).into()
        } else {
            // SAFETY: The precondition is checked in the initial comparison
            // above.
            unsafe {
                #[cfg(feature = "specialization")]
                {
                    <Self as SpecFromSlice<Item>>::spec_from(slice)
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
            Self::from(<Vec<Item, Global> as Like>::Type::from(array))
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

impl<Item, const INLINE: usize> From<<Vec<Item, Global> as Like>::Type>
    for SmallVec<Item, INLINE, Global>
{
    fn from(array: <Vec<Item, Global> as Like>::Type) -> Self {
        if array.capacity() == 0 {
            return Self::new();
        }

        if Self::IS_ZST {
            // "Move" elements to stack buffer. They're ZST so we don't actually
            // have to do anything. Just make sure they're not
            // dropped. We don't wrap the vector in ManuallyDrop so
            // that when it's dropped, the memory is deallocated, if
            // it needs to be.
            let mut vec = array;
            let length = vec.len();

            // SAFETY: `0` is less than the vector's capacity.
            // old_len..new_len is an empty range. So there are no uninitialized
            // elements
            unsafe { vec.set_len(0) };
            Self {
                length: LocatedLength::new(length, false),
                raw: RawSmallVec::new(),
                allocator: Global
            }
        } else {
            let mut vec = ManuallyDrop::new(array);
            let length = vec.len();

            // A heap-allocated `SmallVec` must always observe the invariant
            // that `cap > INLINE`.
            if vec.capacity() <= INLINE {
                vec.reserve(INLINE + 1 - length);
            }
            let cap = vec.capacity();
            // SAFETY: vec.capacity is not `0` (checked above), so the pointer
            // can not dangle and thus specifically cannot be null.
            let ptr = unsafe { NonNull::new_unchecked(vec.as_mut_ptr()) };

            Self {
                length: LocatedLength::new(length, true),
                raw: RawSmallVec::heap(ptr, cap),
                allocator: Global
            }
        }
    }
}

impl<Item, const INLINE: usize, Heap: Allocator> From<SmallVec<Item, INLINE, Heap>>
    for <Vec<Item, Global> as Like>::Type
{
    fn from(this: SmallVec<Item, INLINE, Heap>) -> Self {
        let (length, on_heap) = this.length.parts();
        if !on_heap {
            let mut vec = <Vec<Item, Global> as Like>::Type::with_capacity(length);
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
                <Vec<Item, Global> as Like>::Type::from_raw_parts(ptr.as_ptr(), length, cap)
            }
        }
    }
}

impl<Item, const INLINE: usize, Heap: Allocator> From<SmallVec<Item, INLINE, Heap>>
    for <Box<[Item], Global> as Like>::Type
{
    fn from(this: SmallVec<Item, INLINE, Heap>) -> Self {
        <Vec<Item, Global> as Like>::Type::from(this).into_boxed_slice()
    }
}

impl<T, const N: usize> From<<Box<[T], Global> as Like>::Type> for SmallVec<T, N> {
    fn from(boxed: <Box<[T], Global> as Like>::Type) -> Self {
        boxed.into_vec().into()
    }
}

#[cfg(feature = "arrayvec")]
impl<Item, const M: usize, const INLINE: usize> From<SmallVec<Item, INLINE, Global>>
    for arrayvec::ArrayVec<Item, M>
{
    fn from(this: SmallVec<Item, INLINE, Global>) -> Self {
        // SAFETY: caller ensures length matches M
        <arrayvec::ArrayVec<Item, M> as TryFrom<SmallVec<Item, INLINE, Global>>>::try_from(this)
            .unwrap()
    }
}
