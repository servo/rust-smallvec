use {
    super::{
        Allocator,
        SmallVec
    },
    malloc_size_of::{
        MallocShallowSizeOf,
        MallocSizeOf,
        MallocSizeOfOps
    }
};

impl<T, const N: usize, A: Allocator> MallocShallowSizeOf for SmallVec<T, N, A> {
    fn shallow_size_of(&self, ops: &mut MallocSizeOfOps) -> usize {
        self.spilled()
            .then(|| unsafe { ops.malloc_size_of(self.as_ptr()) })
            .unwrap_or_default()
    }
}

impl<T: MallocSizeOf, const N: usize, A: Allocator> MallocSizeOf for SmallVec<T, N, A> {
    fn size_of(&self, ops: &mut MallocSizeOfOps) -> usize {
        self.iter().map(|item| item.size_of(ops)).sum::<usize>() + self.shallow_size_of(ops)
    }
}
