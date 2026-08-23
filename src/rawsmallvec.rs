<<<<<<< HEAD
use {
    super::{allocationerror::AllocationError, TaggedLen},
    alloc::alloc::Layout,
    core::{
        mem::{ManuallyDrop, MaybeUninit},
        ptr::{copy_nonoverlapping, NonNull},
    },
};
=======
use core::mem::{ManuallyDrop, MaybeUninit};
use core::ptr::NonNull;
use super::TaggedLen;
use super::allocationerror::AllocationError;
use core::ptr::copy_nonoverlapping;
use alloc::alloc::Layout;
>>>>>>> 7fa0992 (feat: added bytes, serde, std, taggedlen files)

/// Either a stack array with `length <= N` or a heap array
/// whose pointer and capacity are stored here.
///
/// We store a `NonNull<T>` instead of a `*mut T` so that type is covariant
/// with respect to `T`, and since the heap pointer is never null.
#[repr(C)]
pub union RawSmallVec<T, const N: usize> {
    pub inline: ManuallyDrop<MaybeUninit<[T; N]>>,
    pub heap: (NonNull<T>, usize),
}

impl<T, const N: usize> RawSmallVec<T, N> {
    pub const IS_ZST: bool = size_of::<T>() == 0;
<<<<<<< HEAD
=======

>>>>>>> 7fa0992 (feat: added bytes, serde, std, taggedlen files)
    #[inline]
    pub const fn new() -> Self {
        Self::new_inline(MaybeUninit::uninit())
    }
    #[inline]
    pub const fn new_inline(inline: MaybeUninit<[T; N]>) -> Self {
        Self {
            inline: ManuallyDrop::new(inline),
        }
    }
    #[inline]
    pub const fn new_heap(ptr: NonNull<T>, capacity: usize) -> Self {
        Self {
            heap: (ptr, capacity),
        }
    }
<<<<<<< HEAD
=======

>>>>>>> 7fa0992 (feat: added bytes, serde, std, taggedlen files)
    #[inline]
    pub const fn as_ptr_inline(&self) -> *const T {
        // SAFETY: it is safe because we aren't reading the value, just getting a
        // reference to it. reading it would be UB potentially, but for that downstream
        // unsafe is required
        #[allow(unused_unsafe, reason = "requires unsafe in MSRV")]
        (unsafe { &raw const self.inline }).cast::<T>()
    }
<<<<<<< HEAD
=======

>>>>>>> 7fa0992 (feat: added bytes, serde, std, taggedlen files)
    #[inline]
    pub const fn as_mut_ptr_inline(&mut self) -> *mut T {
        // SAFETY: same as above
        #[allow(unused_unsafe, reason = "requires unsafe in MSRV")]
        (unsafe { &raw mut self.inline }).cast::<T>()
    }
<<<<<<< HEAD
=======

>>>>>>> 7fa0992 (feat: added bytes, serde, std, taggedlen files)
    /// # Safety
    ///
    /// The vector must be on the heap
    #[inline]
    pub const unsafe fn as_ptr_heap(&self) -> *const T {
        self.heap.0.as_ptr()
    }
<<<<<<< HEAD
=======

>>>>>>> 7fa0992 (feat: added bytes, serde, std, taggedlen files)
    /// # Safety
    ///
    /// The vector must be on the heap
    #[inline]
    pub const unsafe fn as_mut_ptr_heap(&mut self) -> *mut T {
        self.heap.0.as_ptr()
    }
<<<<<<< HEAD
=======

>>>>>>> 7fa0992 (feat: added bytes, serde, std, taggedlen files)
    /// # Safety
    ///
    /// `new_capacity` must be non zero, and greater or equal to the length.
    /// T must not be a ZST.
    pub unsafe fn try_grow_raw(
        &mut self,
        len: TaggedLen<T>,
        new_capacity: usize,
    ) -> Result<(), AllocationError> {
        use alloc::alloc::{alloc, realloc};
        debug_assert!(!Self::IS_ZST);
        debug_assert!(new_capacity > 0);
        debug_assert!(new_capacity >= len.value());
<<<<<<< HEAD
=======

>>>>>>> 7fa0992 (feat: added bytes, serde, std, taggedlen files)
        let was_on_heap = len.on_heap();
        let ptr = if was_on_heap {
            self.as_mut_ptr_heap()
        } else {
            self.as_mut_ptr_inline()
        };
        let len = len.value();
<<<<<<< HEAD
=======

>>>>>>> 7fa0992 (feat: added bytes, serde, std, taggedlen files)
        let new_layout =
            Layout::array::<T>(new_capacity).map_err(|_| AllocationError::CapacityOverflow)?;
        if new_layout.size() > isize::MAX as usize {
            return Err(AllocationError::CapacityOverflow);
        }
<<<<<<< HEAD
=======

>>>>>>> 7fa0992 (feat: added bytes, serde, std, taggedlen files)
        let new_ptr = if len == 0 || !was_on_heap {
            // get a fresh allocation
            let new_ptr = alloc(new_layout) as *mut T; // `new_layout` has nonzero size.
            let new_ptr =
                NonNull::new(new_ptr).ok_or(AllocationError::Failure { layout: new_layout })?;
            copy_nonoverlapping(ptr, new_ptr.as_ptr(), len);
            new_ptr
        } else {
            // use realloc
<<<<<<< HEAD
=======

>>>>>>> 7fa0992 (feat: added bytes, serde, std, taggedlen files)
            // this can't overflow since we already constructed an equivalent layout during
            // the previous allocation
            let old_layout =
                Layout::from_size_align_unchecked(self.heap.1 * size_of::<T>(), align_of::<T>());
<<<<<<< HEAD
=======

>>>>>>> 7fa0992 (feat: added bytes, serde, std, taggedlen files)
            // SAFETY: ptr was allocated with this allocator
            // old_layout is the same as the layout used to allocate the previous memory
            // block new_layout.size() is greater than zero
            // does not overflow when rounded up to alignment. since it was constructed
            // with Layout::array
            let new_ptr = realloc(ptr as *mut u8, old_layout, new_layout.size()) as *mut T;
            NonNull::new(new_ptr).ok_or(AllocationError::Failure { layout: new_layout })?
        };
        *self = Self::new_heap(new_ptr, new_capacity);
        Ok(())
    }
<<<<<<< HEAD
}
=======
}
>>>>>>> 7fa0992 (feat: added bytes, serde, std, taggedlen files)
