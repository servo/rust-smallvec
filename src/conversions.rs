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
        if slice.len() <= Self::inline_size() {
            // SAFETY: The precondition is already checked above
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
        } else {
            <Vec<Item, Global> as Like>::Type::from(slice).into()
        }
    }
}

impl<Item: Clone, const INLINE: usize> From<&mut [Item]> for SmallVec<Item, INLINE, Global> {
    #[inline]
    fn from(slice: &mut [Item]) -> Self {
        Self::from(slice as &[Item])
    }
}

impl<Item: Clone, const INLINE: usize, const LENGTH: usize> From<&[Item; LENGTH]>
    for SmallVec<Item, INLINE, Global>
{
    #[inline]
    fn from(slice: &[Item; LENGTH]) -> Self {
        Self::from(slice as &[Item])
    }
}

impl<Item: Clone, const INLINE: usize, const LENGTH: usize> From<&mut [Item; LENGTH]>
    for SmallVec<Item, INLINE, Global>
{
    #[inline]
    fn from(slice: &mut [Item; LENGTH]) -> Self {
        Self::from(slice as &[Item])
    }
}

impl<Item, const INLINE: usize, const LENGTH: usize> From<[Item; LENGTH]>
    for SmallVec<Item, INLINE, Global>
{
    fn from(array: [Item; LENGTH]) -> Self {
        if const { LENGTH > INLINE } {
            <Vec<Item, Global> as Like>::Type::from(array).into()
        } else {
            let mut smallvec = Self::new();
            let data = ManuallyDrop::new(array);
            // SAFETY: LENGTH <= INLINE, thus it fits
            unsafe {
                copy_nonoverlapping(data.as_ptr(), smallvec.as_mut_ptr(), LENGTH);
                smallvec.set_len(LENGTH);
            }
            smallvec
        }
    }
}
impl<Item, const INLINE: usize, const LENGTH: usize, Heap: Allocator>
    TryFrom<SmallVec<Item, INLINE, Heap>> for [Item; LENGTH]
{
    type Error = SmallVec<Item, INLINE, Heap>;

    #[inline]
    fn try_from(
        mut smallvec: SmallVec<Item, INLINE, Heap>
    ) -> Result<[Item; LENGTH], SmallVec<Item, INLINE, Heap>> {
        // SAFETY: we release ownership of the elements we hold
        (smallvec.len() == LENGTH)
            .then(|| unsafe {
                smallvec.set_len(0);
                smallvec.as_ptr().cast::<[Item; LENGTH]>().read()
            })
            .ok_or(smallvec)
    }
}

impl<Item, const INLINE: usize, Heap: Allocator> From<<Vec<Item, Heap> as Like>::Type>
    for SmallVec<Item, INLINE, Heap>
{
    fn from(vec: <Vec<Item, Heap> as Like>::Type) -> Self {
        let (pointer, length, allocator) = <Vec<Item, Heap> as Like>::destruct(vec);
        if Self::IS_ZST {
            Self {
                length: LocatedLength::new(length, false),
                raw: RawSmallVec::new(),
                allocator
            }
        } else if let Some(pointer) = pointer {
            Self {
                length: LocatedLength::new(length, true),
                raw: RawSmallVec::heap(pointer.cast(), pointer.len()),
                allocator
            }
        } else {
            Self {
                length: LocatedLength::new(length, false),
                raw: RawSmallVec::new(),
                allocator
            }
        }
    }
}
impl<Item, const INLINE: usize, Heap: Allocator> From<SmallVec<Item, INLINE, Heap>>
    for <Vec<Item, Heap> as Like>::Type
{
    fn from(smallvec: SmallVec<Item, INLINE, Heap>) -> Self {
        let (length, on_heap) = smallvec.length.parts();
        let smallvec = ManuallyDrop::new(smallvec);
        if !on_heap {
            let mut vector = <Vec<Item, Heap> as Like>::construct(None, 0, unsafe {
                (&raw const smallvec.allocator).read()
            });
            // SAFETY: we create a new vector with sufficient capacity, copy our
            // elements into it to transfer ownership and then set
            // the length we don't drop the elements we previously
            // held
            unsafe {
                copy_nonoverlapping(smallvec.raw.as_ptr_inline(), vector.as_mut_ptr(), length);
                vector.set_len(length);
            }
            vector
        } else {
            // SAFETY:
            // - `ptr` was created with the SmallVec's allocator
            // - `ptr` was created with the appropriate alignment for `Item`
            // - the allocation pointed to by ptr is exactly cap * sizeof(Item)
            // - `length` is less than or equal to `cap`
            // - the first `length` entries are proper `Item`-values
            // - the allocation is not larger than `isize::MAX`
            unsafe {
                let (ptr, cap) = smallvec.raw.heap;
                Vec::construct(
                    Some(NonNull::slice_from_raw_parts(ptr.cast(), cap)),
                    length,
                    (&raw const smallvec.allocator).read()
                )
            }
        }
    }
}

impl<Item, const INLINE: usize, Heap: Allocator> From<SmallVec<Item, INLINE, Heap>>
    for <Box<[Item], Heap> as Like>::Type
{
    fn from(smallvec: SmallVec<Item, INLINE, Heap>) -> Self {
        <Vec<Item, Heap> as Like>::Type::from(smallvec).into_boxed_slice()
    }
}
impl<Item, const INLINE: usize, Heap: Allocator> From<<Box<[Item], Heap> as Like>::Type>
    for SmallVec<Item, INLINE, Heap>
{
    fn from(boxed: <Box<[Item], Heap> as Like>::Type) -> Self {
        boxed.into_vec().into()
    }
}
