use {
    super::{
        CollectionAllocErr,
        taggedlen::TaggedLen
    },
    alloc::alloc::{
        alloc,
        realloc
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

/// Either a stack array with `length <= N` or a heap array
/// whose pointer and capacity are stored here.
///
/// We store a `NonNull<T>` instead of a `*mut T` so that type is covariant
/// with respect to `T`, and since the heap pointer is never null.
#[repr(C)]
pub union RawSmallVec<T, const N: usize> {
    pub inline: ManuallyDrop<[MaybeUninit<T>; N]>,
    pub heap: NonNull<[MaybeUninit<T>]>
}

impl<T, const N: usize> Default for RawSmallVec<T, N> {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl<T, const N: usize> RawSmallVec<T, N> {
    pub const INLINE_CAP: usize = if Self::IS_ZST { usize::MAX } else { N };
    const IS_ZST: bool = size_of::<T>() == 0;

    #[inline]
    pub const fn new() -> Self {
        Self::new_inline([const { MaybeUninit::uninit() }; N])
    }

    #[inline]
    pub const fn new_inline(inline: [MaybeUninit<T>; N]) -> Self {
        Self {
            inline: ManuallyDrop::new(inline)
        }
    }

    #[inline]
    pub const fn new_heap(ptr: NonNull<[MaybeUninit<T>]>) -> Self {
        Self {
            heap: ptr
        }
    }

    /// # Safety
    ///
    /// `inline` must be the active variant
    /// otherwise it reads pointer as elements
    #[inline]
    pub const unsafe fn as_inline(&self) -> &[MaybeUninit<T>; N] {
        // SAFETY: it is safe because we aren't reading the value, just getting
        // a reference to it. reading it would be UB potentially, but
        // for that downstream unsafe is required
        unsafe {
            (&raw const self.inline)
                .cast::<[MaybeUninit<T>; N]>()
                .as_ref()
                .unwrap_unchecked()
        }
    }

    /// # Safety
    ///
    /// `inline` must be the active variant
    /// otherwise it reads pointer as elements
    #[inline]
    pub const unsafe fn as_mut_inline(&mut self) -> &mut [MaybeUninit<T>; N] {
        // SAFETY: same as above
        unsafe {
            (&raw mut self.inline)
                .cast::<[MaybeUninit<T>; N]>()
                .as_mut()
                .unwrap_unchecked()
        }
    }

    /// # Safety
    ///
    /// `heap` must be the active variant
    /// otherwise it reads inlined elements as pointer
    #[inline]
    pub const unsafe fn as_heap(&self) -> &[MaybeUninit<T>] {
        unsafe { self.heap.as_ref() }
    }

    /// # Safety
    ///
    /// `heap` must be the active variant
    /// otherwise it reads inlined elements as pointer
    #[inline]
    pub const unsafe fn as_mut_heap(&mut self) -> &mut [MaybeUninit<T>] {
        unsafe { self.heap.as_mut() }
    }

    /// # Safety
    ///
    /// `on_heap` must be true if and only if `self.heap` is the active member.
    #[inline(always)]
    pub const unsafe fn capacity(&self, on_heap: bool) -> usize {
        if on_heap {
            unsafe { self.as_heap().len() }
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
        let ptr = if was_on_heap {
            unsafe { self.as_mut_heap() }
        } else {
            unsafe { self.as_mut_inline() }
        }
        .as_mut_ptr();

        let new_layout =
            Layout::array::<T>(new_capacity).map_err(|_| CollectionAllocErr::CapacityOverflow)?;
        if new_layout.size() > isize::MAX as usize {
            return Err(CollectionAllocErr::CapacityOverflow);
        }

        let new_ptr = if !was_on_heap {
            // get a fresh allocation

            let new_ptr = unsafe { alloc(new_layout) } as *mut MaybeUninit<T>;
            let new_ptr = NonNull::new(new_ptr).ok_or(CollectionAllocErr::AllocErr {
                layout: new_layout
            })?;
            unsafe { copy_nonoverlapping(ptr.cast(), new_ptr.as_ptr(), len) };
            new_ptr
        } else {
            // use realloc

            // this can't overflow since we already constructed an equivalent
            // layout during the previous allocation
            let old_layout = unsafe {
                Layout::from_size_align_unchecked(self.heap.len() * size_of::<T>(), align_of::<T>())
            };

            // SAFETY: ptr was allocated with this allocator
            // old_layout is the same as the layout used to allocate the
            // previous memory block new_layout.size() is greater
            // than zero does not overflow when rounded up to
            // alignment. since it was constructed
            // with Layout::array
            let new_ptr = unsafe { realloc(ptr.cast(), old_layout, new_layout.size()) }
                as *mut MaybeUninit<T>;
            NonNull::new(new_ptr).ok_or(CollectionAllocErr::AllocErr {
                layout: new_layout
            })?
        };
        *self = Self::new_heap(NonNull::slice_from_raw_parts(new_ptr, len));
        Ok(())
    }
}
