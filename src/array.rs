// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

/// Types that can be used as the backing store for a SmallVec
pub unsafe trait Array {
    /// The type of the array's elements.
    type Item;
    /// Returns the number of items the array can hold.
    const LEN: usize;
}

macro_rules! impl_array(
    ($($len:expr),+) => {
        $(
            unsafe impl<T> Array for [T; $len] {
                type Item = T;
                const LEN: usize = $len;
            }
        )+
    }
);

impl_array!(
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 20, 24, 32, 36, 0x40, 0x80, 0x100,
    0x200, 0x400, 0x800, 0x1000, 0x2000, 0x4000, 0x8000, 0x10000, 0x20000, 0x40000, 0x80000,
    0x100000
);
