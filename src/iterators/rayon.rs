//! The code is heavily inspired by `rayon/vec.rs`, since it's exactly what we
//! need, except it's all private

use {
    crate::SmallVec,
    core::{
        mem::take,
        ptr::{
            self,
            drop_in_place
        },
        slice
    },
    rayon::{
        iter::plumbing::{
            Producer,
            UnindexedConsumer,
            bridge_producer_consumer
        },
        prelude::ParallelIterator
    }
};

struct SliceDrain<'a, T>(slice::IterMut<'a, T>);

impl<T> Iterator for SliceDrain<'_, T> {
    type Item = T;

    fn next(&mut self) -> Option<T> {
        self.0.next().map(|val| unsafe { ptr::read(val) })
    }
}

impl<T> DoubleEndedIterator for SliceDrain<'_, T> {
    fn next_back(&mut self) -> Option<T> {
        self.0.next_back().map(|val| unsafe { ptr::read(val) })
    }
}

impl<T> ExactSizeIterator for SliceDrain<'_, T> {
    fn len(&self) -> usize {
        self.0.len()
    }
}

impl<T> Drop for SliceDrain<'_, T> {
    fn drop(&mut self) {
        unsafe { drop_in_place(take(&mut self.0).into_slice()) };
    }
}

struct DrainProducer<'a, T>(&'a mut [T]);

impl<'a, T: Send> Producer for DrainProducer<'a, T> {
    type IntoIter = SliceDrain<'a, T>;
    type Item = T;

    fn into_iter(mut self) -> SliceDrain<'a, T> {
        SliceDrain(take(&mut self.0).iter_mut())
    }

    fn split_at(mut self, index: usize) -> (Self, Self) {
        let (left, right) = take(&mut self.0).split_at_mut(index);

        (DrainProducer(left), DrainProducer(right))
    }
}

impl<T> Drop for DrainProducer<'_, T> {
    fn drop(&mut self) {
        unsafe { drop_in_place(self.0) };
    }
}

impl<T: Send, const N: usize> ParallelIterator for SmallVec<T, N> {
    type Item = T;

    fn drive_unindexed<C: UnindexedConsumer<T>>(mut self, consumer: C) -> C::Result {
        let len = self.len();

        bridge_producer_consumer(
            len,
            DrainProducer(unsafe {
                // SAFETY: set_len(0) is always valid
                // All items will either be passed out or dropped by
                // DrainProducer/SliceDrop, so there shouldn't
                // be any possibility for leakage
                self.set_len(0);

                // SAFETY: set_len didn't deallocate/drop the elements, so they
                // are still valid.
                slice::from_raw_parts_mut(self.as_mut_ptr(), len)
            }),
            consumer
        )
    }
}
