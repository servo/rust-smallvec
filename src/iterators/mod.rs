pub mod extractif;
pub mod intoiter;

use {
    super::SmallVec,
    core::{
        iter::FromIterator,
        slice::{
            Iter,
            IterMut
        }
    }
};

impl<T, const N: usize> FromIterator<T> for SmallVec<T, N> {
    #[inline]
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        #[cfg(feature = "specialization")]
        {
            super::spec_traits::SpecFromIterator::<T, _>::spec_from_iter(iter.into_iter())
        }

        #[cfg(not(feature = "specialization"))]
        {
            Self::from_iter_fallback(iter.into_iter())
        }
    }
}

impl<'a, T, const N: usize> IntoIterator for &'a SmallVec<T, N> {
    type IntoIter = Iter<'a, T>;
    type Item = &'a T;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<'a, T, const N: usize> IntoIterator for &'a mut SmallVec<T, N> {
    type IntoIter = IterMut<'a, T>;
    type Item = &'a mut T;

    fn into_iter(self) -> Self::IntoIter {
        self.iter_mut()
    }
}
