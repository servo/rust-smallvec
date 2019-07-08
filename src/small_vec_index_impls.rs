// FIXME: It isn't possible to use constant generics inside a
// macro. See https://github.com/rust-lang/rust/issues/61574

#[cfg(feature = "const_generics")]
mod impls {
    use crate::SmallVec;
    use core::ops::{Index, IndexMut, Range, RangeFrom, RangeFull, RangeTo};

    impl<T, const N: usize> Index<usize> for SmallVec<T, { N }> {
        type Output = T;
        #[inline]
        fn index(&self, index: usize) -> &T {
            &(&**self)[index]
        }
    }
    impl<T, const N: usize> IndexMut<usize> for SmallVec<T, { N }> {
        #[inline]
        fn index_mut(&mut self, index: usize) -> &mut T {
            &mut (&mut **self)[index]
        }
    }

    impl<T, const N: usize> Index<Range<usize>> for SmallVec<T, { N }> {
        type Output = [T];
        #[inline]
        fn index(&self, index: Range<usize>) -> &[T] {
            &(&**self)[index]
        }
    }
    impl<T, const N: usize> IndexMut<Range<usize>> for SmallVec<T, { N }> {
        #[inline]
        fn index_mut(&mut self, index: Range<usize>) -> &mut [T] {
            &mut (&mut **self)[index]
        }
    }

    impl<T, const N: usize> Index<RangeFrom<usize>> for SmallVec<T, { N }> {
        type Output = [T];
        #[inline]
        fn index(&self, index: RangeFrom<usize>) -> &[T] {
            &(&**self)[index]
        }
    }
    impl<T, const N: usize> IndexMut<RangeFrom<usize>> for SmallVec<T, { N }> {
        #[inline]
        fn index_mut(&mut self, index: RangeFrom<usize>) -> &mut [T] {
            &mut (&mut **self)[index]
        }
    }

    impl<T, const N: usize> Index<RangeTo<usize>> for SmallVec<T, { N }> {
        type Output = [T];
        #[inline]
        fn index(&self, index: RangeTo<usize>) -> &[T] {
            &(&**self)[index]
        }
    }
    impl<T, const N: usize> IndexMut<RangeTo<usize>> for SmallVec<T, { N }> {
        #[inline]
        fn index_mut(&mut self, index: RangeTo<usize>) -> &mut [T] {
            &mut (&mut **self)[index]
        }
    }

    impl<T, const N: usize> Index<RangeFull> for SmallVec<T, { N }> {
        type Output = [T];
        #[inline]
        fn index(&self, index: RangeFull) -> &[T] {
            &(&**self)[index]
        }
    }
    impl<T, const N: usize> IndexMut<RangeFull> for SmallVec<T, { N }> {
        #[inline]
        fn index_mut(&mut self, index: RangeFull) -> &mut [T] {
            &mut (&mut **self)[index]
        }
    }
}

#[cfg(not(feature = "const_generics"))]
mod impls {
    use crate::{Array, SmallVec};
    use core::ops::{Index, IndexMut, Range, RangeFrom, RangeFull, RangeTo};

    macro_rules! impl_index {
        ($index_type: ty, $output_type: ty) => {
            impl<A: Array> Index<$index_type> for SmallVec<A> {
                type Output = $output_type;
                #[inline]
                fn index(&self, index: $index_type) -> &$output_type {
                    &(&**self)[index]
                }
            }

            impl<A: Array> IndexMut<$index_type> for SmallVec<A> {
                #[inline]
                fn index_mut(&mut self, index: $index_type) -> &mut $output_type {
                    &mut (&mut **self)[index]
                }
            }
        };
    }

    impl_index!(usize, A::Item);
    impl_index!(Range<usize>, [A::Item]);
    impl_index!(RangeFrom<usize>, [A::Item]);
    impl_index!(RangeTo<usize>, [A::Item]);
    impl_index!(RangeFull, [A::Item]);
}
