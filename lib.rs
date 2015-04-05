/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

//! Small vectors in various sizes. These store a certain number of elements inline and fall back
//! to the heap for larger allocations.

use std::mem::zeroed as i;
use std::cmp;
use std::fmt;
use std::iter::{IntoIterator, FromIterator};
use std::marker::PhantomData;
use std::mem;
use std::ptr;
use std::slice;

// Generic code for all small vectors

pub trait VecLike<T> {
    fn vec_len(&self) -> usize;
    fn vec_push(&mut self, value: T);

    fn vec_slice_mut<'a>(&'a mut self, start: usize, end: usize) -> &'a mut [T];

    #[inline]
    fn vec_slice_from_mut<'a>(&'a mut self, start: usize) -> &'a mut [T] {
        let len = self.vec_len();
        self.vec_slice_mut(start, len)
    }
}

impl<T> VecLike<T> for Vec<T> {
    #[inline]
    fn vec_len(&self) -> usize {
        self.len()
    }

    #[inline]
    fn vec_push(&mut self, value: T) {
        self.push(value);
    }

    #[inline]
    fn vec_slice_mut<'a>(&'a mut self, start: usize, end: usize) -> &'a mut [T] {
        &mut self[start..end]
    }
}

unsafe fn deallocate<T>(ptr: *mut T, capacity: usize) {
    let _vec: Vec<T> = Vec::from_raw_parts(ptr, 0, capacity);
    // Let it drop.
}


pub struct SmallVecIterator<'a, T: 'a> {
    ptr: *const T,
    end: *const T,
    _lifetime: PhantomData<&'a T>
}

impl<'a,T> Iterator for SmallVecIterator<'a,T> {
    type Item = &'a T;

    #[inline]
    fn next(&mut self) -> Option<&'a T> {
        unsafe {
            if self.ptr == self.end {
                return None
            }
            let old = self.ptr;
            self.ptr = if mem::size_of::<T>() == 0 {
                mem::transmute(self.ptr as usize + 1)
            } else {
                self.ptr.offset(1)
            };
            Some(&*old)
        }
    }
}

impl<'a,T> DoubleEndedIterator for SmallVecIterator<'a,T> {
    #[inline]
    fn next_back(&mut self) -> Option<&'a T> {
        unsafe {
            if self.ptr == self.end {
                return None
            }
            self.end = if mem::size_of::<T>() == 0 {
                mem::transmute(self.end as usize - 1)
            } else {
                self.end.offset(-1)
            };
            Some(mem::transmute(self.end))
        }
    }
}

pub struct SmallVecMutIterator<'a, T: 'a> {
    ptr: *mut T,
    end: *mut T,
    _lifetime: PhantomData<&'a T>,
}

impl<'a,T> Iterator for SmallVecMutIterator<'a,T> {
    type Item = &'a mut T;

    #[inline]
    fn next(&mut self) -> Option<&'a mut T> {
        unsafe {
            if self.ptr == self.end {
                return None
            }
            let old = self.ptr;
            self.ptr = if mem::size_of::<T>() == 0 {
                mem::transmute(self.ptr as usize + 1)
            } else {
                self.ptr.offset(1)
            };
            Some(&mut *old)
        }
    }
}

pub struct SmallVecMoveIterator<'a, T: 'a> {
    allocation: Option<*mut T>,
    cap: usize,
    iter: SmallVecMutIterator<'a,T>,
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

// Concrete implementations

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

            pub fn iter<'a>(&'a self) -> SmallVecIterator<'a,T> {
                SmallVecIterator {
                    ptr: self.begin(),
                    end: self.end(),
                    _lifetime: PhantomData,
                }
            }

            pub fn mut_iter<'a>(&'a mut self) -> SmallVecMutIterator<'a,T> {
                SmallVecMutIterator {
                    ptr: self.begin_mut(),
                    end: self.end_mut(),
                    _lifetime: PhantomData,
                }
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
                    let iter = self.mut_iter();
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
                unsafe {
                    let mut vec: Vec<T> = Vec::with_capacity(new_cap);
                    let new_alloc = vec.as_mut_ptr();
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

            pub fn get<'a>(&'a self, index: usize) -> &'a T {
                if index >= self.len() {
                    self.fail_bounds_check(index)
                }
                unsafe {
                    &*self.begin().offset(index as isize)
                }
            }

            pub fn get_mut<'a>(&'a mut self, index: usize) -> &'a mut T {
                if index >= self.len() {
                    self.fail_bounds_check(index)
                }
                unsafe {
                    &mut *self.begin_mut().offset(index as isize)
                }
            }

            pub fn slice<'a>(&'a self, start: usize, end: usize) -> &'a [T] {
                assert!(start <= end);
                assert!(end <= self.len());
                unsafe {
                    slice::from_raw_parts(self.begin().offset(start as isize), end - start)
                }
            }

            pub fn as_slice<'a>(&'a self) -> &'a [T] {
                self.slice(0, self.len())
            }

            pub fn as_slice_mut<'a>(&'a mut self) -> &'a mut [T] {
                let len = self.len();
                self.slice_mut(0, len)
            }

            pub fn slice_mut<'a>(&'a mut self, start: usize, end: usize) -> &'a mut [T] {
                assert!(start <= end);
                assert!(end <= self.len());
                unsafe {
                    slice::from_raw_parts_mut(self.begin_mut().offset(start as isize), end - start)
                }
            }

            pub fn slice_from_mut<'a>(&'a mut self, start: usize) -> &'a mut [T] {
                let len = self.len();
                self.slice_mut(start, len)
            }

            fn fail_bounds_check(&self, index: usize) {
                panic!("index {} beyond length ({})", index, self.len())
            }
        }

        impl<T> VecLike<T> for $name<T> {
            #[inline]
            fn vec_len(&self) -> usize {
                self.len()
            }

            #[inline]
            fn vec_push(&mut self, value: T) {
                self.push(value);
            }

            #[inline]
            fn vec_slice_mut<'a>(&'a mut self, start: usize, end: usize) -> &'a mut [T] {
                self.slice_mut(start, end)
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
                write!(f, "{:?}", self.as_slice())
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
        assert_eq!(v.as_slice(), &[
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
        assert_eq!(v.as_slice(), &[
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
        assert_eq!(v.as_slice(), &[
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
