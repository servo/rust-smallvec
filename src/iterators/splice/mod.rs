mod allocref;

use {
    super::{
        super::{
            Allocator,
            Global,
            SmallVec
        },
        drain::Drain
    },
    allocref::Allocref
};

pub struct Splice<'a, I: Iterator + 'a, const N: usize, A: Allocator = Global> {
    drain: Drain<'a, I::Item, N, A>,
    replace_with: I
}

impl<'a, I: Iterator + 'a, const N: usize, A: Allocator> Splice<'a, I, N, A> {
    pub(crate) fn new(drain: Drain<'a, I::Item, N, A>, replace_with: I) -> Self {
        Self {
            drain,
            replace_with
        }
    }
}

impl<'a, I, const N: usize, A: Allocator> core::fmt::Debug for Splice<'a, I, N, A>
where
    I: core::fmt::Debug + Iterator + 'a,
    <I as Iterator>::Item: core::fmt::Debug
{
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_tuple("Splice").field(&self.drain).finish()
    }
}

impl<I: Iterator, const N: usize, A: Allocator> Iterator for Splice<'_, I, N, A> {
    type Item = I::Item;

    fn next(&mut self) -> Option<Self::Item> {
        self.drain.next()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.drain.size_hint()
    }
}

impl<I: Iterator, const N: usize, A: Allocator> DoubleEndedIterator for Splice<'_, I, N, A> {
    fn next_back(&mut self) -> Option<Self::Item> {
        self.drain.next_back()
    }
}

impl<I: Iterator, const N: usize, A: Allocator> ExactSizeIterator for Splice<'_, I, N, A> {}

impl<I: Iterator, const N: usize, A: Allocator> Drop for Splice<'_, I, N, A> {
    fn drop(&mut self) {
        self.drain.by_ref().for_each(drop);
        // At this point draining is done and the only remaining tasks are
        // splicing and moving things into the final place.
        // Which means we can replace the slice::Iter with pointers that won't
        // point to deallocated memory, so that Drain::drop is still
        // allowed to call iter.len(), otherwise it would break the
        // ptr.sub_ptr contract.
        self.drain.iter = [].iter();

        unsafe {
            if self.drain.tail_len == 0 {
                self.drain.vec.as_mut().extend(self.replace_with.by_ref());
                return;
            }

            // First fill the range left by drain().
            if !self.drain.fill(&mut self.replace_with) {
                return;
            }

            // There may be more elements. Use the lower bound as an estimate.
            // FIXME: Is the upper bound a better guess? Or something else?
            let (lower_bound, _upper_bound) = self.replace_with.size_hint();
            if lower_bound > 0 {
                self.drain.move_tail(lower_bound);
                if !self.drain.fill(&mut self.replace_with) {
                    return;
                }
            }

            let mut smallvec =
                SmallVec::<_, N, _>::new_in(Allocref(&self.drain.vec.as_ref().allocator));
            smallvec.extend(self.replace_with.by_ref());
            let mut collected = smallvec.into_iter();

            // Now we have an exact count.
            if collected.len() > 0 {
                self.drain.move_tail(collected.len());
                let filled = self.drain.fill(&mut collected);
                debug_assert!(filled);
                debug_assert_eq!(collected.len(), 0);
            }
        }
        // Let `Drain::drop` move the tail back if necessary and restore
        // `vec.length`.
    }
}
