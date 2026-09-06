use {
    crate::SmallVec,
    alloc::{
        borrow::Cow,
        collections::VecDeque,
        vec::Vec
    }
};

macro_rules! __impl_slice_eq1 {
    ([$($vars:tt)*] $lhs:ty, $rhs:ty $(where $ty:ty: $bound:ident)?) => {
        impl<T, U, $($vars)*> PartialEq<$rhs> for $lhs
        where
            T: PartialEq<U>,
            $($ty: $bound)?
        {
            #[inline]
            fn eq(&self, other: &$rhs) -> bool { self[..] == other[..] }
        }
    };
}

__impl_slice_eq1! { [const N: usize, const M: usize] SmallVec<T, M>, SmallVec<U, N> }
__impl_slice_eq1! { [const N: usize, const M: usize] SmallVec<T, M>, [U; N] }
__impl_slice_eq1! { [const N: usize, const M: usize] SmallVec<T, M>, &[U; N] }
__impl_slice_eq1! { [const N: usize] SmallVec<T, N>, [U] }
__impl_slice_eq1! { [const N: usize] SmallVec<T, N>, &[U] }
__impl_slice_eq1! { [const N: usize] SmallVec<T, N>, &mut [U] }
__impl_slice_eq1! { [const N: usize] [T], SmallVec<U, N> }
__impl_slice_eq1! { [const N: usize] &[T], SmallVec<U, N> }
__impl_slice_eq1! { [const N: usize] &mut [T], SmallVec<U, N> }
__impl_slice_eq1! { [const N: usize] Vec<T>, SmallVec<U, N> }
__impl_slice_eq1! { [const N: usize] SmallVec<T, N>, Vec<U> }
__impl_slice_eq1! { [const N: usize] Cow<'_, [T]>, SmallVec<U, N> where T: Clone }
__impl_slice_eq1! { [const N: usize] SmallVec<T, N>, Cow<'_, [U]> where U: Clone }

impl<T, U, const N: usize> PartialEq<SmallVec<U, N>> for VecDeque<T>
where T: PartialEq<U>
{
    #[inline]
    fn eq(&self, other: &SmallVec<U, N>) -> bool {
        let other = other.as_slice();
        if self.len() != other.len() {
            return false;
        }
        let (sa, sb) = self.as_slices();
        let (oa, ob) = other[..].split_at(sa.len());
        sa == oa && sb == ob
    }
}

impl<T, const N: usize> Eq for SmallVec<T, N> where T: Eq {}

impl<T, const N: usize> PartialOrd for SmallVec<T, N>
where T: PartialOrd
{
    #[inline]
    fn partial_cmp(&self, other: &SmallVec<T, N>) -> Option<core::cmp::Ordering> {
        self.as_slice().partial_cmp(other.as_slice())
    }
}

impl<T, const N: usize> Ord for SmallVec<T, N>
where T: Ord
{
    #[inline]
    fn cmp(&self, other: &SmallVec<T, N>) -> core::cmp::Ordering {
        self.as_slice().cmp(other.as_slice())
    }
}
