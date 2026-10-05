use {
    crate::{
        Allocator,
        Global,
        LocatedLength,
        RawSmallVec,
        SmallVec
    },
    alloc::alloc::handle_alloc_error,
    core::{
        alloc::Layout,
        mem::{
            ManuallyDrop,
            MaybeUninit
        },
        ptr::{
            NonNull,
            copy_nonoverlapping
        }
    }
};

impl<T: Clone, const N: usize> From<&[T]> for SmallVec<T, N, Global> {
    #[inline]
    fn from(slice: &[T]) -> Self {
        if slice.len() > Self::inline_size() {
            // Standard Rust vectors are already specialized.
            alloc::vec::Vec::from(slice).into()
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
            alloc::vec::Vec::from(array).into()
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

#[cfg(not(feature = "allocator-api"))]
use Allocator as BaseAllocator;
#[cfg(feature = "allocator-api2")]
use allocator_api2::alloc::Allocator as BaseAllocator;
#[cfg(all(feature = "allocator-api", not(feature = "allocator-api2")))]
use core::alloc::Allocator as BaseAllocator;

trait Parts<T, A> {
    fn into_parts(self) -> (*mut T, usize, usize, A);
    unsafe fn from_parts(ptr: *mut T, length: usize, cap: usize, allocator: A) -> Self;
}

#[cfg(feature = "allocator-api")]
impl<T, A: BaseAllocator> Parts<T, A> for crate::Vec<T, A> {
    fn into_parts(self) -> (*mut T, usize, usize, A) {
        #[cfg(feature = "allocator-api2")]
        {
            self.into_raw_parts_with_alloc()
        }
        #[cfg(not(feature = "allocator-api2"))]
        {
            self.into_raw_parts_with_allocator()
        }
    }

    unsafe fn from_parts(ptr: *mut T, length: usize, cap: usize, allocator: A) -> Self {
        unsafe { crate::Vec::from_raw_parts_in(ptr, length, cap, allocator) }
    }
}

#[cfg(any(not(feature = "allocator-api"), feature = "allocator-api2"))]
impl<T> Parts<T, Global> for alloc::vec::Vec<T> {
    fn into_parts(self) -> (*mut T, usize, usize, Global) {
        let (ptr, length, cap) = self.into_raw_parts();
        (ptr, length, cap, Global)
    }

    unsafe fn from_parts(ptr: *mut T, length: usize, cap: usize, _: Global) -> Self {
        unsafe { alloc::vec::Vec::from_raw_parts(ptr, length, cap) }
    }
}

impl<T, const N: usize, A: BaseAllocator> SmallVec<T, N, A> {
    fn from_vec_helper(vec: impl Parts<T, A>) -> Self {
        let (ptr, length, cap, allocator) = vec.into_parts();

        // SAFETY: A `Vec` always has a non-null pointer.
        let ptr = unsafe { NonNull::new_unchecked(ptr) };

        if N < cap {
            Self {
                length: LocatedLength::new(length, !Self::IS_ZST),
                raw: RawSmallVec {
                    heap: (ptr, cap)
                },
                allocator
            }
        } else {
            let mut inline = MaybeUninit::uninit();
            // SAFETY: vec.capacity() <= N
            unsafe {
                copy_nonoverlapping(ptr.as_ptr(), &raw mut inline as *mut T, length);
                // We have to manually deallocate vec's memory, since we need to
                // move its allocator out, meaning we can't drop it
                allocator.deallocate(
                    ptr.cast(),
                    Layout::from_size_align_unchecked(cap * size_of::<T>(), align_of::<T>())
                );
            }
            Self {
                length: LocatedLength::new(length, false),
                raw: RawSmallVec {
                    inline: ManuallyDrop::new(inline)
                },
                allocator
            }
        }
    }

    fn into_vec_helper<P: Parts<T, A>>(self) -> P {
        let (length, on_heap) = self.length.parts();
        let this = ManuallyDrop::new(self);
        unsafe {
            // SAFETY: we don't call any allocator-using methods on `this`, so
            //         it's fine to copy it out
            let allocator = core::ptr::read(&this.allocator);
            if !on_heap {
                let layout =
                    Layout::from_size_align_unchecked(length * size_of::<T>(), align_of::<T>());
                let ptr = Allocator::allocate(&allocator, layout)
                    .unwrap_or_else(|| handle_alloc_error(layout))
                    .as_ptr() as *mut T;

                copy_nonoverlapping(this.raw.as_ptr_inline(), ptr, length);

                P::from_parts(ptr, length, length, allocator)
            } else {
                // SAFETY:
                // - `ptr` was created with the SmallVec's allocator
                // - `ptr` was created with the appropriate alignment for `T`
                // - the allocation pointed to by ptr is exactly cap * sizeof(T)
                // - `length` is less than or equal to `cap`
                // - the first `length` entries are proper `T`-values
                // - the allocation is not larger than `isize::MAX`
                let (ptr, cap) = this.raw.heap;
                P::from_parts(ptr.as_ptr(), length, cap, allocator)
            }
        }
    }
}

#[cfg(feature = "allocator-api")]
impl<T, const N: usize, A: BaseAllocator> From<crate::Vec<T, A>> for SmallVec<T, N, A> {
    fn from(vec: crate::Vec<T, A>) -> Self {
        Self::from_vec_helper(vec)
    }
}

#[cfg(any(not(feature = "allocator-api"), feature = "allocator-api2"))]
impl<T, const N: usize> From<alloc::vec::Vec<T>> for SmallVec<T, N, Global> {
    fn from(vec: alloc::vec::Vec<T>) -> Self {
        Self::from_vec_helper(vec)
    }
}

#[cfg(feature = "allocator-api")]
impl<T, const N: usize, A: BaseAllocator> From<SmallVec<T, N, A>> for crate::Vec<T, A> {
    fn from(this: SmallVec<T, N, A>) -> Self {
        this.into_vec_helper()
    }
}

#[cfg(any(not(feature = "allocator-api"), feature = "allocator-api2"))]
impl<T, const N: usize> From<SmallVec<T, N, Global>> for alloc::vec::Vec<T> {
    fn from(this: SmallVec<T, N, Global>) -> Self {
        this.into_vec_helper()
    }
}

#[cfg(feature = "allocator-api")]
impl<T, const N: usize, A: BaseAllocator> From<SmallVec<T, N, A>>
    for crate::allocator::Box<[T], A>
{
    fn from(this: SmallVec<T, N, A>) -> Self {
        crate::Vec::from(this).into_boxed_slice()
    }
}

#[cfg(any(not(feature = "allocator-api"), feature = "allocator-api2"))]
impl<T, const N: usize> From<SmallVec<T, N, Global>> for alloc::boxed::Box<[T]> {
    fn from(this: SmallVec<T, N, Global>) -> Self {
        alloc::vec::Vec::from(this).into_boxed_slice()
    }
}
