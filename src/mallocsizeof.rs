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

impl<Item, const INLINE: usize, Heap: Allocator> MallocShallowSizeOf
    for SmallVec<Item, INLINE, Heap>
{
    fn shallow_size_of(&self, ops: &mut MallocSizeOfOps) -> usize {
        self.spilled()
            .then(|| unsafe { ops.malloc_size_of(self.as_ptr()) })
            .unwrap_or_default()
    }
}

impl<Item: MallocSizeOf, const INLINE: usize, Heap: Allocator> MallocSizeOf
    for SmallVec<Item, INLINE, Heap>
{
    fn size_of(&self, ops: &mut MallocSizeOfOps) -> usize {
        self.shallow_size_of(ops) + self.iter().map(|item| item.size_of(ops)).sum::<usize>()
    }
}
