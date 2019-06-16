#[cfg(not(feature = "const_generics"))]
use crate::Array;
use core::{mem::MaybeUninit, ptr::NonNull};

#[cfg(feature = "union")]
pub union SmallVecData<A: Array> {
    inline: MaybeUninit<A>,
    heap: (NonNull<A::Item>, usize),
}

#[cfg(feature = "union")]
impl<A: Array> SmallVecData<A> {
    #[inline]
    pub unsafe fn inline(&self) -> &A {
        &*self.inline.as_ptr()
    }
    #[inline]
    pub unsafe fn inline_mut(&mut self) -> &mut A {
        &mut *self.inline.as_mut_ptr()
    }
    #[inline]
    pub fn from_inline(inline: MaybeUninit<A>) -> SmallVecData<A> {
        SmallVecData { inline }
    }
    #[inline]
    pub unsafe fn into_inline(self) -> A {
        self.inline.assume_init()
    }
    #[inline]
    pub unsafe fn heap(&self) -> (*mut A::Item, usize) {
        (self.heap.0.as_ptr(), self.heap.1)
    }
    #[inline]
    pub unsafe fn heap_mut(&mut self) -> (*mut A::Item, &mut usize) {
        (self.heap.0.as_ptr(), &mut self.heap.1)
    }
    #[inline]
    pub fn from_heap(ptr: *mut A::Item, len: usize) -> SmallVecData<A> {
        SmallVecData {
            heap: (NonNull::new(ptr).unwrap(), len),
        }
    }
}

#[cfg(not(feature = "union"))]
pub enum SmallVecData<A: Array> {
    Inline(MaybeUninit<A>),
    Heap((NonNull<A::Item>, usize)),
}

#[cfg(not(feature = "union"))]
impl<A: Array> SmallVecData<A> {
    #[inline]
    pub unsafe fn inline(&self) -> &A {
        match *self {
            SmallVecData::Inline(ref a) => &*a.as_ptr(),
            _ => debug_unreachable!(),
        }
    }
    #[inline]
    pub unsafe fn inline_mut(&mut self) -> &mut A {
        match *self {
            SmallVecData::Inline(ref mut a) => &mut *a.as_mut_ptr(),
            _ => debug_unreachable!(),
        }
    }
    #[inline]
    pub fn from_inline(inline: MaybeUninit<A>) -> SmallVecData<A> {
        SmallVecData::Inline(inline)
    }
    #[inline]
    pub unsafe fn into_inline(self) -> A {
        match self {
            SmallVecData::Inline(a) => a.assume_init(),
            _ => debug_unreachable!(),
        }
    }
    #[inline]
    pub unsafe fn heap(&self) -> (*mut A::Item, usize) {
        match *self {
            SmallVecData::Heap(data) => (data.0.as_ptr(), data.1),
            _ => debug_unreachable!(),
        }
    }
    #[inline]
    pub unsafe fn heap_mut(&mut self) -> (*mut A::Item, &mut usize) {
        match *self {
            SmallVecData::Heap(ref mut data) => (data.0.as_ptr(), &mut data.1),
            _ => debug_unreachable!(),
        }
    }
    #[inline]
    pub fn from_heap(ptr: *mut A::Item, len: usize) -> SmallVecData<A> {
        SmallVecData::Heap((NonNull::new(ptr).unwrap(), len))
    }
}

unsafe impl<A: Array + Send> Send for SmallVecData<A> {}
unsafe impl<A: Array + Sync> Sync for SmallVecData<A> {}
