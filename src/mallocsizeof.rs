use {
    super::SmallVec,
    malloc_size_of::{MallocShallowSizeOf, MallocSizeOf, MallocSizeOfOps},
};
impl<T, const N: usize> MallocShallowSizeOf for SmallVec<T, N> {
    fn shallow_size_of(&self, ops: &mut MallocSizeOfOps) -> usize {
        if self.spilled() {
            unsafe { ops.malloc_size_of(self.as_ptr()) }
        } else {
            0
        }
    }
}
impl<T: MallocSizeOf, const N: usize> MallocSizeOf for SmallVec<T, N> {
    fn size_of(&self, ops: &mut MallocSizeOfOps) -> usize {
        let mut n = self.shallow_size_of(ops);
        for elem in self.iter() {
            n += elem.size_of(ops);
        }
        n
    }
}
