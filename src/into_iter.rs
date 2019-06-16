#[cfg(not(feature = "const_generics"))]
use crate::Array;
use crate::SmallVec;
use core::ptr;

macro_rules! create_with_parts {
(
    <$($({$s_impl_ty_prefix:ident})? $s_impl_ty:ident$(: $s_impl_ty_bound:ident)?),*>,
    <$s_decl_ty:ident$(, {$s_decl_const_ty:ident})?>,
    $array_item:ty
) => {

/// An iterator that consumes a `SmallVec` and yields its items by value.
///
/// Returned from [`SmallVec::into_iter`][1].
///
/// [1]: struct.SmallVec.html#method.into_iter
pub struct IntoIter<$($($s_impl_ty_prefix)? $s_impl_ty$(: $s_impl_ty_bound)?),*> {
    pub(crate) data: SmallVec<$s_decl_ty$(, {$s_decl_const_ty})?>,
    pub(crate) current: usize,
    pub(crate) end: usize,
}

impl<$($($s_impl_ty_prefix)? $s_impl_ty$(: $s_impl_ty_bound)?),*> Drop
    for IntoIter<$s_decl_ty$(, {$s_decl_const_ty})?>
{
    fn drop(&mut self) {
        for _ in self {}
    }
}

impl<$($($s_impl_ty_prefix)? $s_impl_ty$(: $s_impl_ty_bound)?),*> Iterator
    for IntoIter<$s_decl_ty$(, {$s_decl_const_ty})?>
{
    type Item = $array_item;

    #[inline]
    fn next(&mut self) -> Option<$array_item> {
        if self.current == self.end {
            None
        } else {
            unsafe {
                let current = self.current as isize;
                self.current += 1;
                Some(ptr::read(self.data.as_ptr().offset(current)))
            }
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let size = self.end - self.current;
        (size, Some(size))
    }
}

impl<$($($s_impl_ty_prefix)? $s_impl_ty$(: $s_impl_ty_bound)?),*> DoubleEndedIterator
    for IntoIter<$s_decl_ty$(, {$s_decl_const_ty})?>
{
    #[inline]
    fn next_back(&mut self) -> Option<$array_item> {
        if self.current == self.end {
            None
        } else {
            unsafe {
                self.end -= 1;
                Some(ptr::read(self.data.as_ptr().offset(self.end as isize)))
            }
        }
    }
}

impl<$($($s_impl_ty_prefix)? $s_impl_ty$(: $s_impl_ty_bound)?),*> ExactSizeIterator
    for IntoIter<$s_decl_ty$(, {$s_decl_const_ty})?> {}

    }
}

#[cfg(feature = "const_generics")]
create_with_parts!(<T, {const} N: usize>, <T, {N}>, T);
#[cfg(not(feature = "const_generics"))]
create_with_parts!(<A: Array>, <A>, A::Item);
