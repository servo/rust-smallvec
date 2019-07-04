#[cfg(not(feature = "const_generics"))]
use crate::Array;
use core::{mem::MaybeUninit, ptr::NonNull};

macro_rules! create_with_parts {
(
    <$($({$s_impl_ty_prefix:ident})? $s_impl_ty:ident$(: $s_impl_ty_bound:ident)?),*>,
    <$s_decl_ty:ident$(, {$s_decl_const_ty:ident})?>,
    $array:ty,
    $array_item:ty
) => {

#[cfg(feature = "union")]
pub union SmallVecData<$($($s_impl_ty_prefix)? $s_impl_ty$(: $s_impl_ty_bound)?),*> {
    inline: MaybeUninit<$array>,
    heap: (NonNull<$array_item>, usize),
}

#[cfg(feature = "union")]
impl<$($($s_impl_ty_prefix)? $s_impl_ty$(: $s_impl_ty_bound)?),*> SmallVecData<$s_decl_ty$(, {$s_decl_const_ty})?> {
    #[inline]
    pub unsafe fn inline(&self) -> *const $array_item {
        (*self.inline.as_ptr()).as_ptr()
    }
    #[inline]
    pub unsafe fn inline_mut(&mut self) -> *mut $array_item {
        (*self.inline.as_mut_ptr()).as_mut_ptr()
    }
    #[inline]
    pub fn from_inline(inline: MaybeUninit<$array>) -> Self {
        SmallVecData { inline }
    }
    #[inline]
    pub unsafe fn into_inline(self) -> $array {
        self.inline.assume_init()
    }
    #[inline]
    pub unsafe fn heap(&self) -> (*mut $array_item, usize) {
        (self.heap.0.as_ptr(), self.heap.1)
    }
    #[inline]
    pub unsafe fn heap_mut(&mut self) -> (*mut $array_item, &mut usize) {
        (self.heap.0.as_ptr(), &mut self.heap.1)
    }
    #[inline]
    pub fn from_heap(ptr: *mut $array_item, len: usize) -> Self {
        SmallVecData {
            heap: (NonNull::new(ptr).unwrap(), len),
        }
    }
}

#[cfg(not(feature = "union"))]
pub enum SmallVecData<$($($s_impl_ty_prefix)? $s_impl_ty$(: $s_impl_ty_bound)?),*> {
    Inline(MaybeUninit<$array>),
    Heap((NonNull<$array_item>, usize)),
}

#[cfg(not(feature = "union"))]
impl<$($($s_impl_ty_prefix)? $s_impl_ty$(: $s_impl_ty_bound)?),*> SmallVecData<$s_decl_ty$(, {$s_decl_const_ty})?> {
    #[inline]
    pub unsafe fn inline(&self) -> *const $array_item {
        match *self {
            SmallVecData::Inline(ref a) => (*a.as_ptr()).as_ptr(),
            _ => debug_unreachable!(),
        }
    }
    #[inline]
    pub unsafe fn inline_mut(&mut self) -> *mut $array_item {
        match *self {
            SmallVecData::Inline(ref mut a) => (*a.as_mut_ptr()).as_mut_ptr(),
            _ => debug_unreachable!(),
        }
    }
    #[inline]
    pub fn from_inline(inline: MaybeUninit<$array>) -> Self {
        SmallVecData::Inline(inline)
    }
    #[inline]
    pub unsafe fn into_inline(self) -> $array {
        match self {
            SmallVecData::Inline(a) => a.assume_init(),
            _ => debug_unreachable!(),
        }
    }
    #[inline]
    pub unsafe fn heap(&self) -> (*mut $array_item, usize) {
        match *self {
            SmallVecData::Heap(data) => (data.0.as_ptr(), data.1),
            _ => debug_unreachable!(),
        }
    }
    #[inline]
    pub unsafe fn heap_mut(&mut self) -> (*mut $array_item, &mut usize) {
        match *self {
            SmallVecData::Heap(ref mut data) => (data.0.as_ptr(), &mut data.1),
            _ => debug_unreachable!(),
        }
    }
    #[inline]
    pub fn from_heap(ptr: *mut $array_item, len: usize) -> Self {
        SmallVecData::Heap((NonNull::new(ptr).unwrap(), len))
    }
}

unsafe impl<$($($s_impl_ty_prefix)? $s_impl_ty$(: $s_impl_ty_bound)?),*> Send
    for SmallVecData<$s_decl_ty$(, {$s_decl_const_ty})?>
where
    $array_item: Send
{}

unsafe impl<$($($s_impl_ty_prefix)? $s_impl_ty$(: $s_impl_ty_bound)?),*> Sync
    for SmallVecData<$s_decl_ty$(, {$s_decl_const_ty})?>
where
    $array_item: Send
{}

    }
}

#[cfg(feature = "const_generics")]
create_with_parts!(<T, {const} N: usize>, <T, {N}>, [T; N], T);
#[cfg(not(feature = "const_generics"))]
create_with_parts!(<A: Array>, <A>, A, A::Item);
