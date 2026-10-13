//! The code is heavily inspired by `rayon/vec.rs`, since it's exactly what we
//! need, except it's all private

use {
    crate::{
        Allocator,
        SmallVec
    },
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

struct SliceDrain<'a, Item>(slice::IterMut<'a, Item>);

impl<Item> Iterator for SliceDrain<'_, Item> {
    type Item = Item;

    fn next(&mut self) -> Option<Item> {
        self.0.next().map(|val| unsafe { ptr::read(val) })
    }
}

impl<Item> DoubleEndedIterator for SliceDrain<'_, Item> {
    fn next_back(&mut self) -> Option<Item> {
        self.0.next_back().map(|val| unsafe { ptr::read(val) })
    }
}

impl<Item> ExactSizeIterator for SliceDrain<'_, Item> {
    fn len(&self) -> usize {
        self.0.len()
    }
}

impl<Item> Drop for SliceDrain<'_, Item> {
    fn drop(&mut self) {
        unsafe { drop_in_place(take(&mut self.0).into_slice()) };
    }
}

struct DrainProducer<'a, Item>(&'a mut [Item]);

impl<'a, Item: Send> Producer for DrainProducer<'a, Item> {
    type IntoIter = SliceDrain<'a, Item>;
    type Item = Item;

    fn into_iter(mut self) -> SliceDrain<'a, Item> {
        SliceDrain(take(&mut self.0).iter_mut())
    }

    fn split_at(mut self, index: usize) -> (Self, Self) {
        let (left, right) = take(&mut self.0).split_at_mut(index);

        (DrainProducer(left), DrainProducer(right))
    }
}

impl<Item> Drop for DrainProducer<'_, Item> {
    fn drop(&mut self) {
        unsafe { drop_in_place(self.0) };
    }
}

impl<Item: Send, const INLINE: usize, Heap: Allocator + Send> ParallelIterator
    for SmallVec<Item, INLINE, Heap>
{
    type Item = Item;

    fn drive_unindexed<C: UnindexedConsumer<Item>>(mut self, consumer: C) -> C::Result {
        let length = self.len();

        bridge_producer_consumer(
            length,
            DrainProducer(unsafe {
                // SAFETY: set_len(0) is always valid
                // All items will either be passed out or dropped by
                // DrainProducer/SliceDrop, so there shouldn't
                // be any possibility for leakage
                self.set_len(0);

                // SAFETY: set_len didn't deallocate/drop the elements, so they
                // are still valid.
                slice::from_raw_parts_mut(self.as_mut_ptr(), length)
            }),
            consumer
        )
    }
}
