use {
    super::{
        Allocator,
        CollectionAllocErr,
        taggedlen::TaggedLen
    },
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

#[repr(C)]
pub union RawSmallVec<T, const N: usize> {
    pub inline: ManuallyDrop<MaybeUninit<[T; N]>>,
    pub heap: (NonNull<T>, usize)
}

impl<T, const N: usize> RawSmallVec<T, N> {
    pub const INLINE_CAP: usize = if Self::IS_ZST { usize::MAX } else { N };
    const IS_ZST: bool = size_of::<T>() == 0;

    #[inline]
    pub const fn new() -> Self {
        Self::new_inline(MaybeUninit::uninit())
    }

    #[inline]
    pub const fn new_inline(inline: MaybeUninit<[T; N]>) -> Self {
        Self {
            inline: ManuallyDrop::new(inline)
        }
    }

    #[inline]
    pub const fn new_heap(ptr: NonNull<T>, capacity: usize) -> Self {
        Self {
            heap: (ptr, capacity)
        }
    }

    #[inline]
    pub const fn as_ptr_inline(&self) -> *const T {
        // SAFETY: it is safe because we aren't reading the value, just getting
        // a reference to it. reading it would be UB potentially, but
        // for that downstream unsafe is required
        #[allow(unused_unsafe, reason = "Unsafe in MSRV")]
        (unsafe { &raw const self.inline }).cast()
    }

    #[inline]
    pub const fn as_mut_ptr_inline(&mut self) -> *mut T {
        // SAFETY: same as above
        #[allow(unused_unsafe, reason = "Unsafe in MSRV")]
        (unsafe { &raw mut self.inline }).cast()
    }

    /// # Safety
    ///
    /// `on_heap` must be true if and only if `self.heap` is the active member.
    #[inline(always)]
    pub const unsafe fn as_ptr(&self, on_heap: bool) -> *const T {
        if on_heap {
            unsafe { self.heap.0.as_ptr() }
        } else {
            self.as_ptr_inline()
        }
    }

    /// # Safety
    ///
    /// `on_heap` must be true if and only if `self.heap` is the active member.
    #[inline(always)]
    pub const unsafe fn as_mut_ptr(&mut self, on_heap: bool) -> *mut T {
        if on_heap {
            unsafe { self.heap.0.as_ptr() }
        } else {
            self.as_mut_ptr_inline()
        }
    }

    /// # Safety
    ///
    /// `on_heap` must be true if and only if `self.heap` is the active member.
    #[inline(always)]
    pub const unsafe fn capacity(&self, on_heap: bool) -> usize {
        if on_heap {
            unsafe { self.heap.1 }
        } else {
            Self::INLINE_CAP
        }
    }

    /// # Safety
    ///
    /// `new_capacity` must be non zero, and greater or equal to the length.
    /// T must not be a ZST.
    ///
    /// the allocator must be the same one the data was allocated with
    pub unsafe fn try_grow_raw<A: Allocator>(
        &mut self,
        len: TaggedLen<T>,
        new_capacity: usize,
        allocator: &A
    ) -> Result<(), CollectionAllocErr> {
        let (len, was_on_heap) = len.parts();
        debug_assert!(!Self::IS_ZST);
        debug_assert!(new_capacity > 0 && new_capacity >= len);

        // SAFETY: the tag tells which member is active
        let ptr = unsafe { self.as_mut_ptr(was_on_heap) };

        let new_layout =
            Layout::array::<T>(new_capacity).map_err(|_| CollectionAllocErr::CapacityOverflow)?;
        if new_layout.size() > isize::MAX as usize {
            return Err(CollectionAllocErr::CapacityOverflow);
        }

        let new_ptr = if !was_on_heap {
            // get a fresh allocation
            // `new_layout` has nonzero size.
            let new_ptr = allocator
                .allocate(new_layout)
                .map_err(|_| CollectionAllocErr::AllocErr {
                    layout: new_layout
                })?
                .cast();
            unsafe { copy_nonoverlapping(ptr, new_ptr.as_ptr(), len) };
            new_ptr
        } else {
            // use grow

            // this can't overflow since we already constructed an equivalent
            // layout during the previous allocation
            let old_layout = unsafe {
                Layout::from_size_align_unchecked(self.heap.1 * size_of::<T>(), align_of::<T>())
            };

            // SAFETY: ptr was allocated with this allocator
            // old_layout is the same as the layout used to allocate the
            // previous memory block new_layout.size() is greater
            // than zero does not overflow when rounded up to
            // alignment. since it was constructed
            // with Layout::array
            unsafe {
                (if self.heap.1 < new_capacity {
                    A::grow
                } else {
                    A::shrink
                })(
                    allocator,
                    NonNull::new(ptr as *mut u8).unwrap(),
                    old_layout,
                    new_layout
                )
            }
            .map_err(|_| CollectionAllocErr::AllocErr {
                layout: new_layout
            })?
            .cast()
        };
        self.heap = (new_ptr, new_capacity);
        Ok(())
    }
}

impl<Type, const N: usize> Default for RawSmallVec<Type, N> {
    fn default() -> Self {
        Self::new()
    }
}
