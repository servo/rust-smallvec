// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use crate::array::Array;
use crate::tagged_union::TaggedUnion2;
use core::mem::MaybeUninit;

pub struct SmallVec<A: Array> {
    tagged_union: TaggedUnion2<InlineData<A>, HeapData<A>>,
}

pub(crate) struct InlineData<A: Array> {
    pub storage: MaybeUninit<A>,
}

pub(crate) struct HeapData<A: Array> {
    pub capacity: usize,
    pub ptr: core::ptr::NonNull<A::Item>,
}

impl<A: Array> InlineData<A> {
    pub(crate) fn as_ptr(&self) -> *const A::Item {
        self.storage.as_ptr() as _
    }

    pub(crate) fn as_mut_ptr(&mut self) -> *mut A::Item {
        self.storage.as_mut_ptr() as _
    }
}

const LEN_MASK: usize = core::usize::MAX >> 1;
const IS_HEAP_MASK: usize = !LEN_MASK;

pub(crate) const MAX_LEN: usize = LEN_MASK;

impl<A: Array> SmallVec<A> {
    pub fn new() -> Self {
        let mut tagged_union = TaggedUnion2::new_uninit();
        tagged_union.set_tag(0); // len = 0, is_heap = false
        Self { tagged_union }
    }

    pub(crate) unsafe fn set_inline_tag(&mut self, len: usize) -> &mut InlineData<A> {
        assert!((len & LEN_MASK) == len, "overflow");
        self.tagged_union.set_tag(len);
        &mut *self.tagged_union.payload_mut_unchecked::<InlineData<A>>()
    }

    pub(crate) unsafe fn set_heap_tag(&mut self, len: usize) -> &mut HeapData<A> {
        assert!((len & LEN_MASK) == len, "overflow");
        self.tagged_union.set_tag(IS_HEAP_MASK | len);
        &mut *self.tagged_union.payload_mut_unchecked::<HeapData<A>>()
    }

    pub unsafe fn set_len(&mut self, new_len: usize) {
        assert!((new_len & LEN_MASK) == new_len, "overflow");
        let is_heap = self.tagged_union.tag() & IS_HEAP_MASK;
        self.tagged_union.set_tag(is_heap | new_len)
    }

    pub fn len(&self) -> usize {
        self.tagged_union.tag() & LEN_MASK
    }

    fn is_inline(&self) -> bool {
        (self.tagged_union.tag() & IS_HEAP_MASK) == 0
    }

    pub(crate) fn as_inline(&self) -> Result<&InlineData<A>, &HeapData<A>> {
        unsafe {
            if self.is_inline() {
                Ok(&*self.tagged_union.payload_unchecked::<InlineData<A>>())
            } else {
                Err(&*self.tagged_union.payload_unchecked::<HeapData<A>>())
            }
        }
    }

    pub(crate) fn as_inline_mut(&mut self) -> Result<&mut InlineData<A>, &mut HeapData<A>> {
        unsafe {
            if self.is_inline() {
                Ok(&mut *self.tagged_union.payload_mut_unchecked::<InlineData<A>>())
            } else {
                Err(&mut *self.tagged_union.payload_mut_unchecked::<HeapData<A>>())
            }
        }
    }
}
