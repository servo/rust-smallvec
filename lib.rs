/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

//! Small vectors in various sizes. These store a certain number of elements inline and fall back
//! to the heap for larger allocations.

use std::mem::zeroed as i;
use std::cmp;
use std::fmt;
use std::iter::{IntoIterator, FromIterator};
use std::mem;
use std::ops;
use std::ptr;
use std::slice;

// Generic code for all small vectors

pub trait VecLike<T>:
        ops::Index<usize, Output=T> +
        ops::IndexMut<usize> +
        ops::Index<ops::Range<usize>, Output=[T]> +
        ops::IndexMut<ops::Range<usize>> +
        ops::Index<ops::RangeFrom<usize>, Output=[T]> +
        ops::IndexMut<ops::RangeFrom<usize>> +
        ops::Index<ops::RangeTo<usize>, Output=[T]> +
        ops::IndexMut<ops::RangeTo<usize>> +
        ops::Index<ops::RangeFull, Output=[T]> +
        ops::IndexMut<ops::RangeFull> +
        ops::Deref +
        ops::DerefMut {

    fn len(&self) -> usize;
    fn push(&mut self, value: T);
}

impl<T> VecLike<T> for Vec<T> {
    #[inline]
    fn len(&self) -> usize {
        Vec::len(self)
    }

    #[inline]
    fn push(&mut self, value: T) {
        Vec::push(self, value);
    }
}

unsafe fn deallocate<T>(ptr: *mut T, capacity: usize) {
    let _vec: Vec<T> = Vec::from_raw_parts(ptr, 0, capacity);
    // Let it drop.
}

pub struct SmallVecMoveIterator<'a, T: 'a> {
    allocation: Option<*mut T>,
    cap: usize,
    iter: slice::IterMut<'a,T>,
}

impl<'a, T: 'a> Iterator for SmallVecMoveIterator<'a,T> {
    type Item = T;

    #[inline]
    fn next(&mut self) -> Option<T> {
        match self.iter.next() {
            None => None,
            Some(reference) => {
                unsafe {
                    // Zero out the values as we go so they don't get double-freed.
                    Some(mem::replace(reference, mem::zeroed()))
                }
            }
        }
    }
}

impl<'a, T: 'a> Drop for SmallVecMoveIterator<'a,T> {
    fn drop(&mut self) {
        // Destroy the remaining elements.
        for _ in self.by_ref() {}

        match self.allocation {
            None => {}
            Some(allocation) => {
                unsafe {
                    deallocate(allocation, self.cap)
                }
            }
        }
    }
}


macro_rules! impl_index {
    ($name: ident, $index_type: ty, $output_type: ty) => {
        impl<T> ops::Index<$index_type> for $name<T> {
            type Output = $output_type;
            #[inline]
            fn index(&self, index: $index_type) -> &$output_type {
                &(&*self)[index]
            }
        }

        impl<T> ops::IndexMut<$index_type> for $name<T> {
            #[inline]
            fn index_mut(&mut self, index: $index_type) -> &mut $output_type {
                &mut (&mut *self)[index]
            }
        }
    }
}

macro_rules! def_small_vector(
    ($name:ident, $size:expr) => (
        pub struct $name<T> {
            len: usize,
            cap: usize,
            ptr: *const T,
            data: [T; $size],
        }

        impl<T> $name<T> {
            unsafe fn set_len(&mut self, new_len: usize) {
                self.len = new_len
            }
            unsafe fn set_cap(&mut self, new_cap: usize) {
                self.cap = new_cap
            }
            fn data(&self, index: usize) -> *const T {
                let ptr: *const T = &self.data[index];
                ptr
            }
            unsafe fn ptr(&self) -> *const T {
                self.ptr
            }
            unsafe fn mut_ptr(&mut self) -> *mut T {
                self.ptr as *mut T
            }
            unsafe fn set_ptr(&mut self, new_ptr: *mut T) {
                self.ptr = new_ptr as *const T
            }

            pub fn inline_size(&self) -> usize {
                $size
            }
            pub fn len(&self) -> usize {
                self.len
            }
            pub fn is_empty(&self) -> bool {
                self.len == 0
            }
            pub fn cap(&self) -> usize {
                self.cap
            }

            pub fn spilled(&self) -> bool {
                self.cap() > self.inline_size()
            }

            pub fn begin(&self) -> *const T {
                unsafe {
                    if self.spilled() {
                        self.ptr()
                    } else {
                        self.data(0)
                    }
                }
            }

            pub fn begin_mut(&mut self) -> *mut T {
                self.begin() as *mut T
            }

            pub fn end(&self) -> *const T {
                unsafe {
                    self.begin().offset(self.len() as isize)
                }
            }

            pub fn end_mut(&mut self) -> *mut T {
                self.end() as *mut T
            }

            /// NB: For efficiency reasons (avoiding making a second copy of the inline elements), this
            /// actually clears out the original array instead of moving it.
            pub fn into_iter<'a>(&'a mut self) -> SmallVecMoveIterator<'a,T> {
                unsafe {
                    let ptr_opt = if self.spilled() {
                        Some(self.mut_ptr())
                    } else {
                        None
                    };
                    let cap = self.cap();
                    let inline_size = self.inline_size();
                    self.set_cap(inline_size);
                    self.set_len(0);
                    let iter = self.iter_mut();
                    SmallVecMoveIterator {
                        allocation: ptr_opt,
                        cap: cap,
                        iter: iter,
                    }
                }
            }

            pub fn push(&mut self, value: T) {
                let cap = self.cap();
                if self.len() == cap {
                    self.grow(cmp::max(cap * 2, 1))
                }
                let end = self.end_mut();
                unsafe {
                    ptr::write(end, value);
                    let len = self.len();
                    self.set_len(len + 1)
                }
            }

            pub fn push_all_move<V: IntoIterator<Item=T>>(&mut self, other: V) {
                for value in other {
                    self.push(value)
                }
            }

            pub fn pop(&mut self) -> Option<T> {
                if self.len() == 0 {
                    return None
                }
                let last_index = self.len() - 1;
                if (last_index as isize) < 0 {
                    panic!("overflow")
                }
                unsafe {
                    let end_ptr = self.begin_mut().offset(last_index as isize);
                    let value = ptr::replace(end_ptr, mem::uninitialized());
                    self.set_len(last_index);
                    Some(value)
                }
            }

            pub fn grow(&mut self, new_cap: usize) {
                let mut vec: Vec<T> = Vec::with_capacity(new_cap);
                let new_alloc = vec.as_mut_ptr();
                unsafe {
                    mem::forget(vec);
                    ptr::copy_nonoverlapping(self.begin(), new_alloc, self.len());

                    if self.spilled() {
                        deallocate(self.mut_ptr(), self.cap())
                    } else {
                        ptr::write_bytes(self.begin_mut(), 0, self.len())
                    }

                    self.set_ptr(new_alloc);
                    self.set_cap(new_cap)
                }
            }
        }

        impl<T> ops::Deref for $name<T> {
            type Target = [T];
            #[inline]
            fn deref(&self) -> &[T] {
                unsafe {
                    slice::from_raw_parts(self.begin(), self.len())
                }
            }
        }

        impl<T> ops::DerefMut for $name<T> {
            #[inline]
            fn deref_mut(&mut self) -> &mut [T] {
                unsafe {
                    slice::from_raw_parts_mut(self.begin_mut(), self.len())
                }
            }
        }

        impl_index!($name, usize, T);
        impl_index!($name, ops::Range<usize>, [T]);
        impl_index!($name, ops::RangeFrom<usize>, [T]);
        impl_index!($name, ops::RangeTo<usize>, [T]);
        impl_index!($name, ops::RangeFull, [T]);

        impl<T> VecLike<T> for $name<T> {
            #[inline]
            fn len(&self) -> usize {
                $name::len(self)
            }

            #[inline]
            fn push(&mut self, value: T) {
                $name::push(self, value);
            }
        }

        impl<T> FromIterator<T> for $name<T> {
            fn from_iter<I: IntoIterator<Item=T>>(iterable: I) -> $name<T> {
                let mut v = $name::new();

                let iter = iterable.into_iter();
                let (lower_size_bound, _) = iter.size_hint();

                if lower_size_bound > v.cap() {
                    v.grow(lower_size_bound);
                }

                for elem in iter {
                    v.push(elem);
                }

                v
            }
        }

        impl<T> $name<T> {
            pub fn extend<I: Iterator<Item=T>>(&mut self, iter: I) {
                let (lower_size_bound, _) = iter.size_hint();

                let target_len = self.len() + lower_size_bound;

                if target_len > self.cap() {
                   self.grow(target_len);
                }

                for elem in iter {
                    self.push(elem);
                }
            }
        }

        impl<T: fmt::Debug> fmt::Debug for $name<T> {
            fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
                write!(f, "{:?}", &**self)
            }
        }

        impl<T> $name<T> {
            #[inline]
            pub fn new() -> $name<T> {
                unsafe {
                    $name {
                        len: 0,
                        cap: $size,
                        ptr: ptr::null(),
                        data: mem::zeroed(),
                    }
                }
            }
        }

        impl<T> Drop for $name<T> {
            fn drop(&mut self) {
                if !self.spilled() {
                    return
                }

                unsafe {
                    let ptr = self.mut_ptr();
                    for i in 0 .. self.len() {
                        *ptr.offset(i as isize) = mem::uninitialized();
                    }

                    deallocate(self.mut_ptr(), self.cap())
                }
            }
        }

        impl<T: Clone> Clone for $name<T> {
            fn clone(&self) -> $name<T> {
                let mut new_vector = $name::new();
                for element in self.iter() {
                    new_vector.push((*element).clone())
                }
                new_vector
            }
        }

        unsafe impl<T: Send> Send for $name<T> {}
    )
);

def_small_vector!(SmallVec1, 1);
def_small_vector!(SmallVec2, 2);
def_small_vector!(SmallVec4, 4);
def_small_vector!(SmallVec8, 8);
def_small_vector!(SmallVec16, 16);
def_small_vector!(SmallVec24, 24);
def_small_vector!(SmallVec32, 32);

#[cfg(test)]
pub mod tests {
    use SmallVec2;
    use SmallVec16;
    use std::borrow::ToOwned;

    // We heap allocate all these strings so that double frees will show up under valgrind.

    #[test]
    pub fn test_inline() {
        let mut v = SmallVec16::new();
        v.push("hello".to_owned());
        v.push("there".to_owned());
        assert_eq!(&*v, &[
            "hello".to_owned(),
            "there".to_owned(),
        ][..]);
    }

    #[test]
    pub fn test_spill() {
        let mut v = SmallVec2::new();
        v.push("hello".to_owned());
        v.push("there".to_owned());
        v.push("burma".to_owned());
        v.push("shave".to_owned());
        assert_eq!(&*v, &[
            "hello".to_owned(),
            "there".to_owned(),
            "burma".to_owned(),
            "shave".to_owned(),
        ][..]);
    }

    #[test]
    pub fn test_double_spill() {
        let mut v = SmallVec2::new();
        v.push("hello".to_owned());
        v.push("there".to_owned());
        v.push("burma".to_owned());
        v.push("shave".to_owned());
        v.push("hello".to_owned());
        v.push("there".to_owned());
        v.push("burma".to_owned());
        v.push("shave".to_owned());
        assert_eq!(&*v, &[
            "hello".to_owned(),
            "there".to_owned(),
            "burma".to_owned(),
            "shave".to_owned(),
            "hello".to_owned(),
            "there".to_owned(),
            "burma".to_owned(),
            "shave".to_owned(),
        ][..]);
    }
}
