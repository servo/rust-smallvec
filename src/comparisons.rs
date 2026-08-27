use super::SmallVec;

impl<T: PartialEq<U>, U, const N: usize, const M: usize> PartialEq<SmallVec<U, M>>
    for SmallVec<T, N>
{
    fn eq(&self, other: &SmallVec<U, M>) -> bool { self.as_slice().eq(other.as_slice()) }
}
impl<T, const N: usize> Eq for SmallVec<T, N> where T: Eq {}

impl<T: PartialEq<U>, U, const N: usize, const M: usize> PartialEq<[U; M]> for SmallVec<T, N> {
    fn eq(&self, other: &[U; M]) -> bool { self[..] == other[..] }
}

impl<T: PartialEq<U>, U, const N: usize, const M: usize> PartialEq<&[U; M]> for SmallVec<T, N> {
    fn eq(&self, other: &&[U; M]) -> bool { self[..] == other[..] }
}

impl<T: PartialEq<U>, U, const N: usize> PartialEq<[U]> for SmallVec<T, N> {
    fn eq(&self, other: &[U]) -> bool { self[..] == other[..] }
}

impl<T: PartialEq<U>, U, const N: usize> PartialEq<&[U]> for SmallVec<T, N> {
    fn eq(&self, other: &&[U]) -> bool { self[..] == other[..] }
}

impl<T: PartialEq<U>, U, const N: usize> PartialEq<&mut [U]> for SmallVec<T, N> {
    fn eq(&self, other: &&mut [U]) -> bool { self[..] == other[..] }
}

impl<T: PartialOrd, const N: usize> PartialOrd for SmallVec<T, N> {
    fn partial_cmp(&self, other: &SmallVec<T, N>) -> Option<core::cmp::Ordering> {
        self.as_slice().partial_cmp(other.as_slice())
    }
}

impl<T: Ord, const N: usize> Ord for SmallVec<T, N> {
    fn cmp(&self, other: &SmallVec<T, N>) -> core::cmp::Ordering {
        self.as_slice().cmp(other.as_slice())
    }
}
