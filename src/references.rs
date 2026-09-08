use {
    super::{
        Allocator,
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

impl<T, const N: usize, A: Allocator> Borrow<[T]> for SmallVec<T, N, A> {
    #[inline]
    fn borrow(&self) -> &[T] {
        self.as_slice()
    }
}
impl<T, const N: usize, A: Allocator> BorrowMut<[T]> for SmallVec<T, N, A> {
    #[inline]
    fn borrow_mut(&mut self) -> &mut [T] {
        self.as_mut_slice()
    }
}

impl<T, const N: usize, A: Allocator> AsRef<[T]> for SmallVec<T, N, A> {
    #[inline]
    fn as_ref(&self) -> &[T] {
        self.as_slice()
    }
}
impl<T, const N: usize, A: Allocator> AsMut<[T]> for SmallVec<T, N, A> {
    #[inline]
    fn as_mut(&mut self) -> &mut [T] {
        self.as_mut_slice()
    }
}

impl<T, const N: usize, A: Allocator> Deref for SmallVec<T, N, A> {
    type Target = [T];

    #[inline]
    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}
impl<T, const N: usize, A: Allocator> DerefMut for SmallVec<T, N, A> {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.as_mut_slice()
    }
}
