use {
    super::SmallVec,
    core::borrow::{Borrow, BorrowMut},
};
impl<T, const N: usize> core::ops::Deref for SmallVec<T, N> {
    type Target = [T];
    #[inline]
    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}
impl<T, const N: usize> core::ops::DerefMut for SmallVec<T, N> {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.as_mut_slice()
    }
}
impl<T, const N: usize> AsRef<[T]> for SmallVec<T, N> {
    #[inline]
    fn as_ref(&self) -> &[T] {
        self.as_slice()
    }
}
impl<T, const N: usize> AsMut<[T]> for SmallVec<T, N> {
    #[inline]
    fn as_mut(&mut self) -> &mut [T] {
        self.as_mut_slice()
    }
}
impl<T, const N: usize> Borrow<[T]> for SmallVec<T, N> {
    #[inline]
    fn borrow(&self) -> &[T] {
        self.as_slice()
    }
}
impl<T, const N: usize> BorrowMut<[T]> for SmallVec<T, N> {
    #[inline]
    fn borrow_mut(&mut self) -> &mut [T] {
        self.as_mut_slice()
    }
}
