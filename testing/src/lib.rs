#![feature(core_intrinsics)]

use smallvec::{SmallVec, Array};
use std::ptr;

#[cfg(not(feature = "likely"))]
macro_rules! likely {
    ($e:expr) => { $e }
}

#[cfg(feature = "likely")]
macro_rules! likely {
    ($e:expr) => { #[allow(unused_unsafe)] { unsafe { std::intrinsics::likely($e) }} }
}

#[cfg(not(feature = "likely"))]
macro_rules! unlikely {
    ($e:expr) => { $e }
}

#[cfg(feature = "likely")]
macro_rules! unlikely {
    ($e:expr) => { #[allow(unused_unsafe)] { unsafe { std::intrinsics::unlikely($e) }} }
}

pub trait Test<A: Array> {
    fn push_light(&mut self, v: A::Item);
}

#[inline(never)]
#[cold]
fn push_light_cold<A: Array>(vec: &mut SmallVec<A>, v: A::Item) {
    if likely!(vec.spilled()) {
        let len = vec.len();
        if likely!(len < vec.capacity()) {
            unsafe {
                ptr::write(vec.as_mut_ptr().offset(len as isize), v);
                vec.set_len(len + 1);
            }
        } else {
            vec.push_old(v)
        }
    } else {
        unsafe {
            std::intrinsics::assume(vec.capacity() == A::size());
        }
        vec.push_old(v)
    }
}

impl<A: Array> Test<A> for SmallVec<A> {
    #[inline(always)]
    fn push_light(&mut self, v: A::Item) {
        let (free, len) = if !self.spilled() {
            let len = self.len();
            (len < A::size(), len)
        } else {
            (false, 0)
        };
        if likely!(free) {
            unsafe {
                ptr::write(self.as_mut_ptr().offset(len as isize), v);
                std::intrinsics::assume(!self.spilled());
                self.set_len(self.len() + 1);
            }
        } else {
            push_light_cold(self, v);
        }
    }
}

// FIXME: Remove
#[no_mangle]
pub fn test1(a: &mut SmallVec<[u32; 8]>) {
    a.push(5);
}

#[no_mangle]
pub fn test2(a: &mut SmallVec<[u32; 8]>) {
    a.push_light(5);
}
