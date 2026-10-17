use {
    super::{
        allocator::Allocator,
        SmallVec
    },
    core::{
        borrow::{
            Borrow,
            BorrowMut
        },
        ops::{
            Deref,
            DerefMut
        }
    }
};

impl<Item, const INLINE: usize, Heap: Allocator> Borrow<[Item]> for SmallVec<Item, INLINE, Heap> {
    #[inline]
    fn borrow(&self) -> &[Item] {
        self.as_ref()
    }
}

impl<Item, const INLINE: usize, Heap: Allocator> BorrowMut<[Item]>
    for SmallVec<Item, INLINE, Heap>
{
    #[inline]
    fn borrow_mut(&mut self) -> &mut [Item] {
        self.as_mut()
    }
}

impl<Item, const INLINE: usize, Heap: Allocator> AsRef<[Item]> for SmallVec<Item, INLINE, Heap> {
    #[inline]
    fn as_ref(&self) -> &[Item] {
        let (length, on_heap) = self.length.parts();
        // SAFETY: all the elements in `..length` are initialized
        unsafe { core::slice::from_raw_parts(self.raw.as_ptr(on_heap), length) }
    }
}

impl<Item, const INLINE: usize, Heap: Allocator> AsMut<[Item]> for SmallVec<Item, INLINE, Heap> {
    #[inline]
    fn as_mut(&mut self) -> &mut [Item] {
        let (length, on_heap) = self.length.parts();
        // SAFETY: see above
        unsafe { core::slice::from_raw_parts_mut(self.raw.as_mut_ptr(on_heap), length) }
    }
}

impl<Item, const INLINE: usize, Heap: Allocator> Deref for SmallVec<Item, INLINE, Heap> {
    type Target = [Item];

    #[inline]
    fn deref(&self) -> &Self::Target {
        self.as_ref()
    }
}

impl<Item, const INLINE: usize, Heap: Allocator> DerefMut for SmallVec<Item, INLINE, Heap> {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.as_mut()
    }
}
