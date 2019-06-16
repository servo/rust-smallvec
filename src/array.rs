/// Types that can be used as the backing store for a SmallVec
pub unsafe trait Array {
    /// The type of the array's elements.
    type Item;
    /// Returns a mutable pointer to the first element of the array.
    fn as_mut_ptr(&mut self) -> *mut Self::Item;
    /// Returns a pointer to the first element of the array.
    fn as_ptr(&self) -> *const Self::Item;
    /// Returns the number of items the array can hold.
    fn size() -> usize;
}

macro_rules! impl_array(
    ($($size:expr),+) => {
        $(
            unsafe impl<T> Array for [T; $size] {
                type Item = T;
                fn as_mut_ptr(&mut self) -> *mut T { self.as_mut().as_mut_ptr() }
                fn as_ptr(&self) -> *const T { self.as_ref().as_ptr() }
                fn size() -> usize { $size }
            }
        )+
    }
);

impl_array!(
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 20, 24, 32, 36, 0x40, 0x80, 0x100,
    0x200, 0x400, 0x800, 0x1000, 0x2000, 0x4000, 0x8000, 0x10000, 0x20000, 0x40000, 0x80000,
    0x100000
);
