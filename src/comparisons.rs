#[cfg(feature = "equivalent")]
use equivalent::{
    Comparable,
    Equivalent
};
#[cfg(feature = "rkyv")]
use rkyv::vec::ArchivedVec;
use {
    super::{
        SmallVec,
        allocator::Allocator
    },
    alloc::{
        borrow::Cow,
        collections::VecDeque,
        vec::Vec
    },
    core::cmp::Ordering
};

macro_rules! implementation {
    ([$($generics:tt)*] for $left:ty =? $right:ty $(where $typename:ty: $bound:ident)?) => {
        impl<
            Item: PartialEq<Other>,
            Other,
            Heap: Allocator,
            const INLINE: usize,
            $($generics)*
        > PartialEq<$right> for $left $(where $typename: $bound)? {
            #[inline]
            fn eq(&self, other: &$right) -> bool {
                *AsRef::<[Item]>::as_ref(self) == *AsRef::<[Other]>::as_ref(other)
            }
        }
    };
    ([$($generics:tt)*] for $left:ty =<> $right:ty $(where $typename:ty: $bound:ident)?) => {
        impl<
            Item: PartialOrd<Other>,
            Other,
            Heap: Allocator,
            const INLINE: usize,
            $($generics)*
        > PartialOrd<$right> for $left $(where $typename: $bound)? {
            #[inline]
            fn partial_cmp(&self, other: &$right) -> Option<Ordering> {
                AsRef::<[Item]>::as_ref(self).iter().partial_cmp(AsRef::<[Other]>::as_ref(other))
            }
        }
    };
    ([$($generics:tt)*] for $left:ty =?+ $right:ty $(where $typename:ty: $bound:ident)?) => {
        impl<
            Item: Equivalent<Other>,
            Other,
            Heap: Allocator,
            const INLINE: usize,
            $($generics)*
        > Equivalent<$right> for $left $(where $typename: $bound)? {
            #[inline]
            fn equivalent(&self, key: &$right) -> bool {
                self.len() == key.len() && self.iter()
                    .zip(AsRef::<[Other]>::as_ref(key))
                    .all(|(item, key)| item.equivalent(key))
            }
        }
    };
    ([$($generics:tt)*] for $left:ty =<>+ $right:ty $(where $typename:ty: $bound:ident)?) => {
        impl<
            Item: Comparable<Other>,
            Other,
            Heap: Allocator,
            const INLINE: usize,
            $($generics)*
        > Comparable<$right> for $left $(where $typename: $bound)? {
            #[inline]
            fn compare(&self, other: &$right) -> Ordering {
                self.iter()
                    .zip(AsRef::<[Other]>::as_ref(other))
                    .map(|(item, key)| item.compare(key))
                    .find(|&ordering| ordering != Ordering::Equal)
                    .unwrap_or_else(|| self.len().cmp(&other.len()))
            }
        }
    }
}

implementation!([const LENGTH: usize, Assigner: Allocator] for SmallVec<Item, INLINE, Heap> =? SmallVec<Other, LENGTH, Assigner>);
impl<Item: Eq, const INLINE: usize, Heap: Allocator> Eq for SmallVec<Item, INLINE, Heap> {}
implementation!([const LENGTH: usize, Assigner: Allocator] for SmallVec<Item, INLINE, Heap> =<> SmallVec<Other, LENGTH, Assigner>);
impl<Item: Ord, const INLINE: usize, Heap: Allocator> Ord for SmallVec<Item, INLINE, Heap> {
    #[inline]
    fn cmp(&self, other: &SmallVec<Item, INLINE, Heap>) -> Ordering {
        self.as_ref().cmp(other.as_ref())
    }
}

implementation!([const LENGTH: usize] for SmallVec<Item, INLINE, Heap> =? [Other; LENGTH]);
implementation!([const LENGTH: usize] for [Item; INLINE] =? SmallVec<Other, LENGTH, Heap>);
#[cfg(feature = "equivalent")]
implementation!([const LENGTH: usize] for SmallVec<Item, INLINE, Heap> =?+ [Other; LENGTH]);
#[cfg(feature = "equivalent")]
implementation!([const LENGTH: usize] for [Item; INLINE] =?+ SmallVec<Other, LENGTH, Heap>);
implementation!([const LENGTH: usize] for SmallVec<Item, INLINE, Heap> =<> [Other; LENGTH]);
implementation!([const LENGTH: usize] for [Item; INLINE] =<> SmallVec<Other, LENGTH, Heap>);
#[cfg(feature = "equivalent")]
implementation!([const LENGTH: usize] for SmallVec<Item, INLINE, Heap> =<>+ [Other; LENGTH]);
#[cfg(feature = "equivalent")]
implementation!([const LENGTH: usize] for [Item; INLINE] =<>+ SmallVec<Other, LENGTH, Heap>);

implementation!([const LENGTH: usize] for SmallVec<Item, INLINE, Heap> =? &[Other; LENGTH]);
implementation!([const LENGTH: usize] for &[Item; INLINE] =? SmallVec<Other, LENGTH, Heap>);
#[cfg(feature = "equivalent")]
implementation!([const LENGTH: usize] for SmallVec<Item, INLINE, Heap> =?+ &[Other; LENGTH]);
#[cfg(feature = "equivalent")]
implementation!([const LENGTH: usize] for &[Item; INLINE] =?+ SmallVec<Other, LENGTH, Heap>);
implementation!([const LENGTH: usize] for SmallVec<Item, INLINE, Heap> =<> &[Other; LENGTH]);
implementation!([const LENGTH: usize] for &[Item; INLINE] =<> SmallVec<Other, LENGTH, Heap>);
#[cfg(feature = "equivalent")]
implementation!([const LENGTH: usize] for SmallVec<Item, INLINE, Heap> =<>+ &[Other; LENGTH]);
#[cfg(feature = "equivalent")]
implementation!([const LENGTH: usize] for &[Item; INLINE] =<>+ SmallVec<Other, LENGTH, Heap>);

implementation!([const LENGTH: usize] for SmallVec<Item, INLINE, Heap> =? &mut [Other; LENGTH]);
implementation!([const LENGTH: usize] for &mut [Item; INLINE] =? SmallVec<Other, LENGTH, Heap>);
#[cfg(feature = "equivalent")]
implementation!([const LENGTH: usize] for SmallVec<Item, INLINE, Heap> =?+ &mut [Other; LENGTH]);
#[cfg(feature = "equivalent")]
implementation!([const LENGTH: usize] for &mut [Item; INLINE] =?+ SmallVec<Other, LENGTH, Heap>);
implementation!([const LENGTH: usize] for SmallVec<Item, INLINE, Heap> =<> &mut [Other; LENGTH]);
implementation!([const LENGTH: usize] for &mut [Item; INLINE] =<> SmallVec<Other, LENGTH, Heap>);
#[cfg(feature = "equivalent")]
implementation!([const LENGTH: usize] for SmallVec<Item, INLINE, Heap> =<>+ &mut [Other; LENGTH]);
#[cfg(feature = "equivalent")]
implementation!([const LENGTH: usize] for &mut [Item; INLINE] =<>+ SmallVec<Other, LENGTH, Heap>);

implementation!([] for SmallVec<Item, INLINE, Heap> =? [Other]);
implementation!([] for [Item] =? SmallVec<Other, INLINE, Heap>);
#[cfg(feature = "equivalent")]
implementation!([] for SmallVec<Item, INLINE, Heap> =?+ [Other]);
implementation!([] for SmallVec<Item, INLINE, Heap> =<> [Other]);
implementation!([] for [Item] =<> SmallVec<Other, INLINE, Heap>);
#[cfg(feature = "equivalent")]
implementation!([] for SmallVec<Item, INLINE, Heap> =<>+ [Other]);

implementation!([] for SmallVec<Item, INLINE, Heap> =? &[Other]);
implementation!([] for &[Item] =? SmallVec<Other, INLINE, Heap>);
#[cfg(feature = "equivalent")]
implementation!([] for SmallVec<Item, INLINE, Heap> =?+ &[Other]);
#[cfg(feature = "equivalent")]
implementation!([] for &[Item] =?+ SmallVec<Other, INLINE, Heap>);
implementation!([] for SmallVec<Item, INLINE, Heap> =<> &[Other]);
implementation!([] for &[Item] =<> SmallVec<Other, INLINE, Heap>);
#[cfg(feature = "equivalent")]
implementation!([] for SmallVec<Item, INLINE, Heap> =<>+ &[Other]);
#[cfg(feature = "equivalent")]
implementation!([] for &[Item] =<>+ SmallVec<Other, INLINE, Heap>);

implementation!([] for SmallVec<Item, INLINE, Heap> =? &mut [Other]);
implementation!([] for &mut [Item] =? SmallVec<Other, INLINE, Heap>);
#[cfg(feature = "equivalent")]
implementation!([] for SmallVec<Item, INLINE, Heap> =?+ &mut [Other]);
#[cfg(feature = "equivalent")]
implementation!([] for &mut [Item] =?+ SmallVec<Other, INLINE, Heap>);
implementation!([] for SmallVec<Item, INLINE, Heap> =<> &mut [Other]);
implementation!([] for &mut [Item] =<> SmallVec<Other, INLINE, Heap>);
#[cfg(feature = "equivalent")]
implementation!([] for SmallVec<Item, INLINE, Heap> =<>+ &mut [Other]);
#[cfg(feature = "equivalent")]
implementation!([] for &mut [Item] =<>+ SmallVec<Other, INLINE, Heap>);

implementation!([] for SmallVec<Item, INLINE, Heap> =? Vec<Other>);
implementation!([] for Vec<Item> =? SmallVec<Other, INLINE, Heap>);
#[cfg(feature = "equivalent")]
implementation!([] for SmallVec<Item, INLINE, Heap> =?+ Vec<Other>);
#[cfg(feature = "equivalent")]
implementation!([] for Vec<Item> =?+ SmallVec<Other, INLINE, Heap>);
implementation!([] for SmallVec<Item, INLINE, Heap> =<> Vec<Other>);
implementation!([] for Vec<Item> =<> SmallVec<Other, INLINE, Heap>);
#[cfg(feature = "equivalent")]
implementation!([] for SmallVec<Item, INLINE, Heap> =<>+ Vec<Other>);
#[cfg(feature = "equivalent")]
implementation!([] for Vec<Item> =<>+ SmallVec<Other, INLINE, Heap>);

#[cfg(feature = "rkyv")]
implementation!([] for ArchivedVec<Item> =? SmallVec<Other, INLINE, Heap>);
#[cfg(feature = "rkyv")]
implementation!([] for SmallVec<Item, INLINE, Heap> =? ArchivedVec<Other>);

implementation!([] for SmallVec<Item, INLINE, Heap> =? Cow<'_, [Other]> where Other: Clone);
implementation!([] for Cow<'_, [Item]> =? SmallVec<Other, INLINE, Heap> where Item: Clone);
#[cfg(feature = "equivalent")]
implementation!([] for SmallVec<Item, INLINE, Heap> =?+ Cow<'_, [Other]> where Other: Clone);
#[cfg(feature = "equivalent")]
implementation!([] for Cow<'_, [Item]> =?+ SmallVec<Other, INLINE, Heap> where Item: Clone);
implementation!([] for SmallVec<Item, INLINE, Heap> =<> Cow<'_, [Other]> where Other: Clone);
implementation!([] for Cow<'_, [Item]> =<> SmallVec<Other, INLINE, Heap> where Item: Clone);
#[cfg(feature = "equivalent")]
implementation!([] for SmallVec<Item, INLINE, Heap> =<>+ Cow<'_, [Other]> where Other: Clone);
#[cfg(feature = "equivalent")]
implementation!([] for Cow<'_, [Item]> =<>+ SmallVec<Other, INLINE, Heap> where Item: Clone);

impl<Item: PartialEq<Other>, Other, const INLINE: usize, Heap: Allocator>
    PartialEq<SmallVec<Other, INLINE, Heap>> for VecDeque<Item>
{
    #[inline]
    fn eq(&self, other: &SmallVec<Other, INLINE, Heap>) -> bool {
        let (first, second) = self.as_slices();
        let Some((before, after)) = other.split_at_checked(first.len()) else {
            return false;
        };
        first == before && second == after
    }
}
impl<Item: PartialEq<Other>, Other, const INLINE: usize, Heap: Allocator> PartialEq<VecDeque<Other>>
    for SmallVec<Item, INLINE, Heap>
{
    #[inline]
    fn eq(&self, other: &VecDeque<Other>) -> bool {
        let (first, second) = other.as_slices();
        let Some((before, after)) = self.split_at_checked(first.len()) else {
            return false;
        };
        before == first && after == second
    }
}
#[cfg(feature = "equivalent")]
impl<Item: Equivalent<Other>, Other, const INLINE: usize, Heap: Allocator>
    Equivalent<SmallVec<Other, INLINE, Heap>> for VecDeque<Item>
{
    #[inline]
    fn equivalent(&self, key: &SmallVec<Other, INLINE, Heap>) -> bool {
        let (first, second) = self.as_slices();
        let Some((before, after)) = key.split_at_checked(first.len()) else {
            return false;
        };
        first.len() == before.len()
            && first
                .iter()
                .zip(before)
                .all(|(item, key)| item.equivalent(key))
            && second.len() == after.len()
            && second
                .iter()
                .zip(after)
                .all(|(item, key)| item.equivalent(key))
    }
}
#[cfg(feature = "equivalent")]
impl<Item: Equivalent<Other>, Other, const INLINE: usize, Heap: Allocator>
    Equivalent<VecDeque<Other>> for SmallVec<Item, INLINE, Heap>
{
    #[inline]
    fn equivalent(&self, key: &VecDeque<Other>) -> bool {
        let (first, second) = key.as_slices();
        let Some((before, after)) = self.split_at_checked(first.len()) else {
            return false;
        };
        first.len() == before.len()
            && first
                .iter()
                .zip(before)
                .all(|(item, key)| key.equivalent(item))
            && second.len() == after.len()
            && second
                .iter()
                .zip(after)
                .all(|(item, key)| key.equivalent(item))
    }
}
impl<Item: PartialOrd<Other>, Other, const INLINE: usize, Heap: Allocator>
    PartialOrd<SmallVec<Other, INLINE, Heap>> for VecDeque<Item>
{
    #[inline]
    fn partial_cmp(&self, other: &SmallVec<Other, INLINE, Heap>) -> Option<Ordering> {
        self.iter().partial_cmp(other.iter())
    }
}
impl<Item: PartialOrd<Other>, Other, const INLINE: usize, Heap: Allocator>
    PartialOrd<VecDeque<Other>> for SmallVec<Item, INLINE, Heap>
{
    #[inline]
    fn partial_cmp(&self, other: &VecDeque<Other>) -> Option<Ordering> {
        self.iter().partial_cmp(other.iter())
    }
}
#[cfg(feature = "equivalent")]
impl<Item: Comparable<Other>, Other, const INLINE: usize, Heap: Allocator>
    Comparable<SmallVec<Other, INLINE, Heap>> for VecDeque<Item>
{
    #[inline]
    fn compare(&self, key: &SmallVec<Other, INLINE, Heap>) -> Ordering {
        let (first, second) = self.as_slices();
        first
            .iter()
            .chain(second.iter())
            .map(|item| item)
            .zip(key.iter())
            .map(|(item, key)| item.compare(key))
            .find(|&ordering| ordering != Ordering::Equal)
            .unwrap_or_else(|| self.len().cmp(&key.len()))
    }
}
#[cfg(feature = "equivalent")]
impl<Item: Comparable<Other>, Other, const INLINE: usize, Heap: Allocator>
    Comparable<VecDeque<Other>> for SmallVec<Item, INLINE, Heap>
{
    #[inline]
    fn compare(&self, key: &VecDeque<Other>) -> Ordering {
        let (first, second) = key.as_slices();
        self.iter()
            .zip(first.iter().chain(second.iter()))
            .map(|(item, key)| item.compare(key))
            .find(|&ordering| ordering != Ordering::Equal)
            .unwrap_or_else(|| self.len().cmp(&key.len()))
    }
}
