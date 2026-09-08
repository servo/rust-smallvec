use {
    super::{
        Allocator,
        CollectionAllocErr,
        infallible,
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
pub(crate) union RawSmallVecInner<T, const N: usize> {
    pub(crate) inline: ManuallyDrop<MaybeUninit<[T; N]>>,
    pub(crate) heap: (NonNull<T>, usize)
}

/// Either a stack array with `length <= N` or a heap array
/// whose pointer and capacity are stored here.
///
/// We store a `NonNull<T>` instead of a `*mut T` so that type is covariant
/// with respect to `T`, and since the heap pointer is never null.
pub struct RawSmallVec<T, const N: usize, A> {
    pub(crate) inner: RawSmallVecInner<T, N>,
    pub(crate) alloc: A
}

impl<T, const N: usize, A: Allocator> RawSmallVec<T, N, A> {
    pub const INLINE_CAP: usize = if Self::IS_ZST { usize::MAX } else { N };
    const IS_ZST: bool = size_of::<T>() == 0;

    #[inline]
    pub const fn new(alloc: A) -> Self {
        Self::new_inline(MaybeUninit::uninit(), alloc)
    }

    #[inline]
    pub const fn new_inline(inline: MaybeUninit<[T; N]>, alloc: A) -> Self {
        Self {
            inner: RawSmallVecInner {
                inline: ManuallyDrop::new(inline)
            },
            alloc
        }
    }

    #[inline]
    pub const fn new_heap(ptr: NonNull<T>, capacity: usize, alloc: A) -> Self {
        Self {
            inner: RawSmallVecInner {
                heap: (ptr, capacity)
            },
            alloc
        }
    }

    #[inline]
    pub const fn as_ptr_inline(&self) -> *const T {
        // SAFETY: it is safe because we aren't reading the value, just getting
        // a reference to it. reading it would be UB potentially, but
        // for that downstream unsafe is required
        #[allow(unused_unsafe, reason = "Unsafe in MSRV")]
        (unsafe { &raw const self.inner.inline }).cast()
    }

    #[inline]
    pub const fn as_mut_ptr_inline(&mut self) -> *mut T {
        // SAFETY: same as above
        #[allow(unused_unsafe, reason = "Unsafe in MSRV")]
        (unsafe { &raw mut self.inner.inline }).cast()
    }

    /// # Safety
    ///
    /// `on_heap` must be true if and only if `self.heap` is the active member.
    #[inline(always)]
    pub const unsafe fn as_ptr(&self, on_heap: bool) -> *const T {
        if on_heap {
            unsafe { self.inner.heap.0.as_ptr() }
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
            unsafe { self.inner.heap.0.as_ptr() }
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
            unsafe { self.inner.heap.1 }
        } else {
            Self::INLINE_CAP
        }
    }

    /// # Safety
    ///
    /// `new_capacity` must be non zero, and greater or equal to the length.
    /// T must not be a ZST.
    pub unsafe fn try_grow_raw(
        &mut self,
        len: TaggedLen<T>,
        new_capacity: usize
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
            let new_ptr = self
                .alloc
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
                Layout::from_size_align_unchecked(
                    self.inner.heap.1 * size_of::<T>(),
                    align_of::<T>()
                )
            };

            // SAFETY: ptr was allocated with this allocator
            // old_layout is the same as the layout used to allocate the
            // previous memory block new_layout.size() is greater
            // than zero does not overflow when rounded up to
            // alignment. since it was constructed
            // with Layout::array
            unsafe {
                self.alloc.grow(
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
        self.inner.heap = (new_ptr, new_capacity);
        Ok(())
    }

    /// # Safety
    ///
    /// `new_capacity` must be non zero, and smaller or equal to the current
    /// one. T must not be a ZST. Items must be stored on the heap.
    pub unsafe fn shrink_to_raw(&mut self, target: usize) {
        unsafe {
            // this can't overflow since it's smaller than one we already made
            let new_layout =
                Layout::from_size_align_unchecked(target * size_of::<T>(), align_of::<T>());

            self.inner.heap = (
                infallible(
                    // SAFETY: ptr was allocated with this allocator
                    // old_layout is the same as the layout used to
                    // allocate the previous
                    // memory block
                    self.alloc
                        .shrink(
                            NonNull::new(self.inner.heap.0.as_ptr() as *mut u8).unwrap(),
                            // this can't overflow since we already constructed an equivalent
                            // layout during the previous allocation
                            Layout::from_size_align_unchecked(
                                self.inner.heap.1 * size_of::<T>(),
                                align_of::<T>()
                            ),
                            new_layout
                        )
                        .map_err(|_| CollectionAllocErr::AllocErr {
                            layout: new_layout
                        })
                )
                .cast(),
                target
            );
        }
    }
}
