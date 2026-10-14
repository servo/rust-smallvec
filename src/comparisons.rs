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
        impl<Item, U, Heap: Allocator, const INLINE: usize, $($vars)*> PartialEq<$rhs> for $lhs
        where
            Item: PartialEq<U>,
            $($ty: $bound)?
        {
            #[inline]
            fn eq(&self, other: &$rhs) -> bool { self[..] == other[..] }
        }
    };
}

__impl_slice_eq1! { [const M: usize, A2: Allocator]
    SmallVec<Item, M, Heap>,
    SmallVec<U, INLINE, A2>
}
__impl_slice_eq1! { [const M: usize] SmallVec<Item, M, Heap>, [U; INLINE] }
__impl_slice_eq1! { [const M: usize] SmallVec<Item, M, Heap>, &[U; INLINE] }
__impl_slice_eq1! { [] SmallVec<Item, INLINE, Heap>, [U] }
__impl_slice_eq1! { [] SmallVec<Item, INLINE, Heap>, &[U] }
__impl_slice_eq1! { [] SmallVec<Item, INLINE, Heap>, &mut [U] }
__impl_slice_eq1! { [] [Item], SmallVec<U, INLINE, Heap> }
__impl_slice_eq1! { [] &[Item], SmallVec<U, INLINE, Heap> }
__impl_slice_eq1! { [] &mut [Item], SmallVec<U, INLINE, Heap> }
__impl_slice_eq1! { [] Vec<Item>, SmallVec<U, INLINE, Heap> }
__impl_slice_eq1! { [] SmallVec<Item, INLINE, Heap>, Vec<U> }
__impl_slice_eq1! { [] Cow<'_, [Item]>, SmallVec<U, INLINE, Heap> where Item: Clone }
__impl_slice_eq1! { [] SmallVec<Item, INLINE, Heap>, Cow<'_, [U]> where U: Clone }

impl<Item, U, const INLINE: usize, Heap: Allocator> PartialEq<SmallVec<U, INLINE, Heap>>
    for VecDeque<Item>
where Item: PartialEq<U>
{
    #[inline]
    fn eq(&self, other: &SmallVec<U, INLINE, Heap>) -> bool {
        let other = other.as_slice();
        if self.len() != other.len() {
            return false;
        }
        let (sa, sb) = self.as_slices();
        let (oa, ob) = other[..].split_at(sa.len());
        sa == oa && sb == ob
    }
}

impl<Item: Eq, const INLINE: usize, Heap: Allocator> Eq for SmallVec<Item, INLINE, Heap> {}

impl<Item, const INLINE: usize, Heap: Allocator> PartialOrd for SmallVec<Item, INLINE, Heap>
where Item: PartialOrd
{
    #[inline]
    fn partial_cmp(&self, other: &SmallVec<Item, INLINE, Heap>) -> Option<core::cmp::Ordering> {
        self.as_slice().partial_cmp(other.as_slice())
    }
}

impl<Item, const INLINE: usize, Heap: Allocator> Ord for SmallVec<Item, INLINE, Heap>
where Item: Ord
{
    #[inline]
    fn cmp(&self, other: &SmallVec<Item, INLINE, Heap>) -> core::cmp::Ordering {
        self.as_slice().cmp(other.as_slice())
    }
}
