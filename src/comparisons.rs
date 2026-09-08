use {
    crate::{
        Allocator,
        SmallVec
    },
    alloc::{
        borrow::Cow,
        collections::VecDeque,
        vec::Vec
    }
};

macro_rules! __impl_slice_eq1 {
    ([$($vars:tt)*] $lhs:ty, $rhs:ty $(where $ty:ty: $bound:ident)?) => {
        impl<T, U, A: Allocator, $($vars)*> PartialEq<$rhs> for $lhs
        where
            T: PartialEq<U>,
            $($ty: $bound)?
        {
            #[inline]
            fn eq(&self, other: &$rhs) -> bool { self[..] == other[..] }
        }
    };
}

__impl_slice_eq1! { [const N: usize, const M: usize] SmallVec<T, M, A>, SmallVec<U, N, A> }
__impl_slice_eq1! { [const N: usize, const M: usize] SmallVec<T, M, A>, [U; N] }
__impl_slice_eq1! { [const N: usize, const M: usize] SmallVec<T, M, A>, &[U; N] }
__impl_slice_eq1! { [const N: usize] SmallVec<T, N, A>, [U] }
__impl_slice_eq1! { [const N: usize] SmallVec<T, N, A>, &[U] }
__impl_slice_eq1! { [const N: usize] SmallVec<T, N, A>, &mut [U] }
__impl_slice_eq1! { [const N: usize] [T], SmallVec<U, N, A> }
__impl_slice_eq1! { [const N: usize] &[T], SmallVec<U, N, A> }
__impl_slice_eq1! { [const N: usize] &mut [T], SmallVec<U, N, A> }
__impl_slice_eq1! { [const N: usize] Vec<T>, SmallVec<U, N, A> }
__impl_slice_eq1! { [const N: usize] SmallVec<T, N, A>, Vec<U> }
__impl_slice_eq1! { [const N: usize] Cow<'_, [T]>, SmallVec<U, N, A> where T: Clone }
__impl_slice_eq1! { [const N: usize] SmallVec<T, N, A>, Cow<'_, [U]> where U: Clone }

impl<T, U, const N: usize, A: Allocator> PartialEq<SmallVec<U, N, A>> for VecDeque<T>
where T: PartialEq<U>
{
    #[inline]
    fn eq(&self, other: &SmallVec<U, N, A>) -> bool {
        let other = other.as_slice();
        if self.len() != other.len() {
            return false;
        }
        let (sa, sb) = self.as_slices();
        let (oa, ob) = other[..].split_at(sa.len());
        sa == oa && sb == ob
    }
}

impl<T, const N: usize, A: Allocator> Eq for SmallVec<T, N, A> where T: Eq {}

impl<T, const N: usize, A: Allocator> PartialOrd for SmallVec<T, N, A>
where T: PartialOrd
{
    #[inline]
    fn partial_cmp(&self, other: &SmallVec<T, N, A>) -> Option<core::cmp::Ordering> {
        self.as_slice().partial_cmp(other.as_slice())
    }
}

impl<T, const N: usize, A: Allocator> Ord for SmallVec<T, N, A>
where T: Ord
{
    #[inline]
    fn cmp(&self, other: &SmallVec<T, N, A>) -> core::cmp::Ordering {
        self.as_slice().cmp(other.as_slice())
    }
}
