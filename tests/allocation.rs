#![cfg(all(
    any(feature = "allocator-api", feature = "allocator-api2"),
    feature = "std" // `std` is necessary for the base `System` allocator
))]

#[cfg(not(feature = "allocator-api2"))]
extern crate alloc;
#[cfg(not(feature = "allocator-api2"))]
use alloc::alloc::{
    AllocError,
    Allocator,
    Layout
};
#[cfg(feature = "allocator-api2")]
use allocator_api2::alloc::{
    AllocError,
    Allocator,
    Layout
};
#[cfg(feature = "allocator-api2")]
use std::alloc::Allocator as _;
use {
    core::{
        cell::{
            Cell,
            RefCell
        },
        ptr::NonNull
    },
    smallvec::{
        SmallVec,
        SmallVecError
    },
    std::{
        alloc::System,
        assert_matches
    }
};

#[derive(Clone, Copy, Debug, Default)]
struct AllocStats {
    pub alloc_calls: usize,
    pub dealloc_calls: usize,
    pub grow_calls: usize,
    pub shrink_calls: usize,
    pub allocated_bytes: isize,
    pub alloc_align: usize,
    pub dealloc_align: usize
}

#[derive(Clone, Debug, Default)]
struct TestAlloc {
    stats: RefCell<AllocStats>
}

impl TestAlloc {
    #[inline]
    fn new() -> Self {
        Self::default()
    }

    #[inline]
    fn stats(&self) -> AllocStats {
        *self.stats.borrow()
    }

    #[inline]
    fn is_clean(&self) -> bool {
        let s = self.stats.borrow();
        s.alloc_calls == s.dealloc_calls && s.allocated_bytes == 0
    }
}

unsafe impl Allocator for TestAlloc {
    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        let ptr = System.allocate(layout).map_err(|_| AllocError)?;
        let mut s = self.stats.borrow_mut();
        s.alloc_calls += 1;
        s.allocated_bytes += layout.size() as isize;
        s.alloc_align = layout.align();
        Ok(ptr)
    }

    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
        let mut s = self.stats.borrow_mut();
        s.dealloc_calls += 1;
        s.allocated_bytes -= layout.size() as isize;
        s.dealloc_align = layout.align();
        unsafe { System.deallocate(ptr, layout) };
    }

    unsafe fn grow(
        &self,
        ptr: NonNull<u8>,
        old: Layout,
        new: Layout
    ) -> Result<NonNull<[u8]>, AllocError> {
        let out = unsafe { System.grow(ptr, old, new) }.map_err(|_| AllocError)?;
        let mut s = self.stats.borrow_mut();
        s.grow_calls += 1;
        s.allocated_bytes += new.size() as isize - old.size() as isize;
        Ok(out)
    }

    unsafe fn shrink(
        &self,
        ptr: NonNull<u8>,
        old: Layout,
        new: Layout
    ) -> Result<NonNull<[u8]>, AllocError> {
        let out = unsafe { System.shrink(ptr, old, new) }.map_err(|_| AllocError)?;
        let mut s = self.stats.borrow_mut();
        s.shrink_calls += 1;
        s.allocated_bytes -= old.size() as isize - new.size() as isize;
        Ok(out)
    }
}

// An allocator that delegates to [`TestAlloc`] but fails once it has handed
// out `remaining` successful allocations. Used to verify the `try_*`
// recovery paths.
#[derive(Clone, Debug, Default)]
struct FailingAlloc {
    inner: TestAlloc,
    remaining: Cell<usize>
}

impl FailingAlloc {
    #[inline]
    fn new(remaining: usize) -> Self {
        Self {
            inner: TestAlloc::new(),
            remaining: Cell::new(remaining)
        }
    }

    #[inline]
    fn stats(&self) -> AllocStats {
        self.inner.stats()
    }

    #[inline]
    fn is_clean(&self) -> bool {
        self.inner.is_clean()
    }
}

unsafe impl Allocator for FailingAlloc {
    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        let remaining = self.remaining.get();
        if remaining == 0 {
            return Err(AllocError);
        }
        self.remaining.set(remaining - 1);
        self.inner.allocate(layout)
    }

    #[inline]
    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
        unsafe { self.inner.deallocate(ptr, layout) }
    }
}

#[test]
fn test_inline_ops_do_not_allocate() {
    const INLINE_SIZE: usize = 8;

    let alloc = TestAlloc::new();

    let mut vec = SmallVec::<i32, INLINE_SIZE, &TestAlloc>::new_in(&alloc);
    vec.extend(0..INLINE_SIZE as i32);
    assert!(!vec.spilled());

    vec.pop();
    vec.push(69);
    vec.remove(2);
    vec.insert(0, 2);
    vec.swap_remove(0);
    vec.fill(420);
    vec.retain(|&x| x % 2 == 0);
    vec.truncate(3);
    drop(vec.drain(..1));
    vec.clear();

    // `grow` within the inline capacity must not allocate
    vec.grow(INLINE_SIZE);

    // `split_off` that stays inline must not allocate either
    vec.extend(0..6);
    let tail = vec.split_off(3);
    assert_eq!(vec.as_slice(), &[0, 1, 2]);
    assert_eq!(tail.as_slice(), &[3, 4, 5]);

    drop(vec);
    drop(tail);

    assert_eq!(alloc.stats().alloc_calls, 0);
    assert_eq!(alloc.stats().dealloc_calls, 0);
    assert!(alloc.is_clean());
}

#[test]
fn test_spill_then_drop() {
    const INLINE_SIZE: usize = 8;

    let alloc = TestAlloc::new();

    let mut vec = SmallVec::<i32, INLINE_SIZE, &TestAlloc>::new_in(&alloc);
    for i in 0..INLINE_SIZE as _ {
        vec.push(i);
    }

    // spill
    vec.push(42);
    // dealloc
    drop(vec);

    assert_eq!(alloc.stats().alloc_calls, 1);
    assert!(alloc.is_clean());
}

#[test]
fn test_with_capacity_in_allocates_properly() {
    const INLINE_SIZE: usize = 4;

    let alloc = TestAlloc::new();

    // allocate eagerly above inline size
    let vec = SmallVec::<i32, INLINE_SIZE, &TestAlloc>::with_capacity_in(INLINE_SIZE + 1, &alloc);
    drop(vec);

    assert_eq!(alloc.stats().alloc_calls, 1);
    assert!(alloc.is_clean());

    let vec = SmallVec::<i32, INLINE_SIZE, &TestAlloc>::with_capacity_in(INLINE_SIZE, &alloc);
    drop(vec);

    assert_eq!(alloc.stats().alloc_calls, 1);
    assert!(alloc.is_clean());
}

#[test]
fn test_moving_does_not_deallocate() {
    const INLINE_SIZE: usize = 4;

    let alloc_a = TestAlloc::new();
    let alloc_b = TestAlloc::new();

    let vec = SmallVec::<i32, INLINE_SIZE, &TestAlloc>::with_capacity_in(INLINE_SIZE + 1, &alloc_a);
    let (ptr, len, cap, allocator) = vec.into_parts_with_alloc();

    // moving to a different inline size must not deallocate
    let vec = unsafe {
        SmallVec::<i32, { INLINE_SIZE + 4 }, &TestAlloc>::from_parts_in(ptr, len, cap, allocator)
    };
    drop(vec);

    // and the deallocation must go through `alloc_a`, not `alloc_b`
    assert_eq!(alloc_a.stats().alloc_calls, 1);
    assert_eq!(alloc_a.stats().dealloc_calls, 1);
    assert!(alloc_a.is_clean());
    assert_eq!(alloc_b.stats().alloc_calls, 0);
    assert_eq!(alloc_b.stats().dealloc_calls, 0);
}

#[test]
fn test_unspill_deallocates() {
    const INLINE_SIZE: usize = 2;

    let alloc = TestAlloc::new();

    let mut vec = SmallVec::<i32, INLINE_SIZE, &TestAlloc>::new_in(&alloc);
    for i in 0..(INLINE_SIZE + 1) as _ {
        vec.push(i);
    }

    assert!(vec.spilled());

    vec.remove(1);
    vec.shrink_to_fit();

    assert!(!vec.spilled());
    assert_eq!(alloc.stats().alloc_calls, 1);
    assert_eq!(alloc.stats().dealloc_calls, 1);

    drop(vec);

    assert_eq!(alloc.stats().dealloc_calls, 1);
    assert!(alloc.is_clean());
}

#[test]
fn test_capacity_overflow_preserves_vec() {
    const INLINE_SIZE: usize = 6;

    let alloc = TestAlloc::new();

    // `try_grow` overflow on a spilled vec
    let mut vec = SmallVec::<i32, INLINE_SIZE, &TestAlloc>::new_in(&alloc);
    for i in 0..(INLINE_SIZE + 1) as _ {
        vec.push(i);
    }
    assert_eq!(alloc.stats().alloc_calls, 1);

    assert_eq!(
        vec.try_grow((isize::MAX as usize) + 4),
        Err(SmallVecError::CapacityOverflow)
    );
    assert_eq!(
        vec.as_slice(),
        core::array::from_fn::<i32, { INLINE_SIZE + 1 }, _>(|i| i as _)
    );
    assert_eq!(alloc.stats().alloc_calls, 1);

    drop(vec);

    // `try_reserve` overflow on an inline vec
    let mut vec = SmallVec::<i32, INLINE_SIZE, &TestAlloc>::new_in(&alloc);
    vec.push(1);
    assert_eq!(
        vec.try_reserve(usize::MAX),
        Err(SmallVecError::CapacityOverflow)
    );
    assert_eq!(vec.as_slice(), &[1]);
    assert_eq!(alloc.stats().alloc_calls, 1);

    drop(vec);
    assert!(alloc.is_clean());
}

#[test]
fn test_no_alloc_on_zst() {
    const INLINE_SIZE: usize = 2;

    let alloc = TestAlloc::new();

    // pushing past the inline capacity must not allocate
    let mut vec = SmallVec::<(), INLINE_SIZE, &TestAlloc>::new_in(&alloc);
    for _ in 0..(INLINE_SIZE + 1) as _ {
        vec.push(());
    }
    drop(vec);

    // neither must requesting a large capacity
    let vec = SmallVec::<(), INLINE_SIZE, &TestAlloc>::with_capacity_in(1024, &alloc);
    assert_eq!(vec.capacity(), usize::MAX);
    drop(vec);

    assert_eq!(alloc.stats().alloc_calls, 0);
    assert_eq!(alloc.stats().dealloc_calls, 0);
    assert!(alloc.is_clean());
}

#[test]
fn test_grow_reuses_allocation() {
    const INLINE_SIZE: usize = 4;

    let alloc = TestAlloc::new();

    let mut vec = SmallVec::<i32, INLINE_SIZE, &TestAlloc>::new_in(&alloc);
    for i in 0..(INLINE_SIZE + 1) as i32 {
        vec.push(i);
    }
    assert_eq!(alloc.stats().alloc_calls, 1);

    // growing an already-spilled vec must realloc, not allocate a fresh block
    vec.reserve(INLINE_SIZE * 4);

    assert_eq!(alloc.stats().alloc_calls, 1);
    assert_eq!(alloc.stats().grow_calls, 1);
    assert_eq!(alloc.stats().dealloc_calls, 0);

    drop(vec);
    assert!(alloc.is_clean());
}

#[test]
fn test_shrink_to_fit_reuses_allocation() {
    const INLINE_SIZE: usize = 4;

    let alloc = TestAlloc::new();

    let mut vec = SmallVec::<i32, INLINE_SIZE, &TestAlloc>::with_capacity_in(64, &alloc);
    vec.extend(0..8); // still above `INLINE_SIZE`

    vec.shrink_to_fit();

    let s = alloc.stats();
    assert_eq!(s.shrink_calls, 1);
    assert_eq!(s.alloc_calls, 1);
    assert_eq!(s.dealloc_calls, 0);

    drop(vec);
    assert!(alloc.is_clean());
}

#[test]
fn test_shrink_to_unspills() {
    const INLINE_SIZE: usize = 4;

    let alloc = TestAlloc::new();

    let mut vec = SmallVec::<i32, INLINE_SIZE, &TestAlloc>::with_capacity_in(16, &alloc);
    vec.extend(0..2);
    assert!(vec.spilled());

    // should be at most 2, so it should inline again.
    vec.shrink_to(0);

    assert!(!vec.spilled());
    assert_eq!(vec.as_slice(), &[0, 1]);
    assert_eq!(alloc.stats().alloc_calls, 1);
    assert_eq!(alloc.stats().dealloc_calls, 1);

    drop(vec);
    assert!(alloc.is_clean());
}

#[test]
fn test_shrink_to_reallocates() {
    const INLINE_SIZE: usize = 4;

    let alloc = TestAlloc::new();

    let mut vec = SmallVec::<i32, INLINE_SIZE, &TestAlloc>::with_capacity_in(64, &alloc);
    vec.extend(0..8);
    assert!(vec.spilled());

    // should be at most 8, so it should remain spilled and shrink.
    vec.shrink_to(8);

    assert!(vec.spilled());
    assert_eq!(vec.capacity(), 8);
    assert_eq!(vec.as_slice(), &[0, 1, 2, 3, 4, 5, 6, 7]);
    assert_eq!(alloc.stats().shrink_calls, 1);
    assert_eq!(alloc.stats().alloc_calls, 1);
    assert_eq!(alloc.stats().dealloc_calls, 0);

    drop(vec);
    assert!(alloc.is_clean());
}

#[test]
fn test_shrink_to_noop() {
    const INLINE_SIZE: usize = 4;

    let alloc = TestAlloc::new();

    let mut vec = SmallVec::<i32, INLINE_SIZE, &TestAlloc>::with_capacity_in(8, &alloc);
    vec.extend(0..8);

    let before = alloc.stats();
    vec.shrink_to(8); // capacity == min, so no-op
    vec.shrink_to(100); // capacity < min, so no-op
    let after = alloc.stats();

    assert_eq!(before.alloc_calls, after.alloc_calls);
    assert_eq!(before.dealloc_calls, after.dealloc_calls);
    assert_eq!(after.shrink_calls, 0);
    assert_eq!(vec.capacity(), 8);

    drop(vec);
    assert!(alloc.is_clean());
}

#[test]
fn test_growth_is_amortized() {
    const INLINE_SIZE: usize = 4;
    const N: usize = 1024;

    let alloc = TestAlloc::new();

    let mut vec = SmallVec::<i32, INLINE_SIZE, &TestAlloc>::new_in(&alloc);
    for i in 0..N as i32 {
        vec.push(i);
    }
    assert_eq!(vec.len(), N);

    // only the initial spill allocates; every later growth reallocates
    assert_eq!(alloc.stats().alloc_calls, 1);
    // doubling growth here means about less than 16 allocations
    assert!(alloc.stats().grow_calls <= 16);

    drop(vec);
    assert!(alloc.is_clean());
}

#[test]
fn test_clone_allocates_and_clone_from_reuses() {
    const INLINE_SIZE: usize = 4;

    let alloc = TestAlloc::new();

    let mut vec = SmallVec::<i32, INLINE_SIZE, &TestAlloc>::new_in(&alloc);
    vec.extend(0..8);
    assert!(vec.spilled());
    assert_eq!(alloc.stats().alloc_calls, 1);

    // cloning a spilled vec allocates
    let clone = vec.clone();
    assert_eq!(clone.as_slice(), vec.as_slice());
    assert_eq!(alloc.stats().alloc_calls, 2);

    // clone_from into a vec with enough capacity must not allocate
    let mut target = SmallVec::<i32, INLINE_SIZE, &TestAlloc>::with_capacity_in(8, &alloc);
    let before = alloc.stats().alloc_calls;
    target.clone_from(&vec);
    assert_eq!(target.as_slice(), vec.as_slice());
    assert_eq!(alloc.stats().alloc_calls, before);

    drop(clone);
    drop(target);
    drop(vec);
    assert!(alloc.is_clean());
}

#[test]
fn test_drop_correctness() {
    use core::cell::Cell;

    struct DropCounter<'a>(&'a Cell<usize>);
    impl Drop for DropCounter<'_> {
        fn drop(&mut self) {
            self.0.set(self.0.get() + 1);
        }
    }

    const INLINE_SIZE: usize = 2;

    let alloc = TestAlloc::new();
    let counter = Cell::new(0);

    {
        let mut vec = SmallVec::<DropCounter<'_>, INLINE_SIZE, &TestAlloc>::new_in(&alloc);
        for _ in 0..6 {
            vec.push(DropCounter(&counter));
        }
        assert!(vec.spilled());
        assert_eq!(counter.get(), 0);

        vec.truncate(4);
        assert_eq!(counter.get(), 2);

        vec.clear();
        assert_eq!(counter.get(), 6);
    }

    assert_eq!(counter.get(), 6);
    assert!(alloc.is_clean());
}

#[test]
fn test_alignment_is_respected() {
    #[repr(align(64))]
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    struct Aligned(u8);

    const INLINE_SIZE: usize = 2;

    let alloc = TestAlloc::new();

    let mut vec = SmallVec::<Aligned, INLINE_SIZE, &TestAlloc>::new_in(&alloc);
    for _ in 0..(INLINE_SIZE + 1) {
        vec.push(Aligned(1));
    }
    assert!(vec.spilled());
    assert_eq!(vec[0].0, 1);

    assert_eq!(alloc.stats().alloc_align, 64);

    drop(vec);
    assert_eq!(alloc.stats().dealloc_align, 64);
    assert!(alloc.is_clean());
}

#[test]
#[should_panic]
fn test_into_parts_with_alloc_inline_panics() {
    const INLINE_SIZE: usize = 4;

    let alloc = TestAlloc::new();

    let vec = SmallVec::<i32, INLINE_SIZE, &TestAlloc>::new_in(&alloc);
    let _ = vec.into_parts_with_alloc();
}

#[test]
fn test_try_with_capacity_in_failure() {
    const INLINE_SIZE: usize = 4;

    let alloc = FailingAlloc::new(0);

    let result = SmallVec::<i32, INLINE_SIZE, &FailingAlloc>::try_with_capacity_in(8, &alloc);
    assert_matches!(result, Err(SmallVecError::AllocationError(_)));
    assert!(alloc.is_clean());
}

#[test]
fn test_allocation_failure_preserves_vec() {
    const INLINE_SIZE: usize = 4;

    // allow the spill, fail the next allocation
    let alloc = FailingAlloc::new(1);

    let mut vec = SmallVec::<i32, INLINE_SIZE, &FailingAlloc>::new_in(&alloc);
    for i in 0..(INLINE_SIZE + 1) as i32 {
        vec.push(i);
    }
    assert!(vec.spilled());
    assert_eq!(alloc.stats().alloc_calls, 1);

    // both entry points must fail without touching the vec
    assert_matches!(vec.try_grow(64), Err(SmallVecError::AllocationError(_)));
    assert_matches!(vec.try_reserve(100), Err(SmallVecError::AllocationError(_)));

    assert_eq!(vec.as_slice(), &[0, 1, 2, 3, 4]);
    assert!(vec.spilled());
    assert_eq!(alloc.stats().alloc_calls, 1);

    drop(vec);
    assert!(alloc.is_clean());
}
