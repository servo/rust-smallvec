use core::ops::{
    Bound,
    Range,
    RangeBounds
};

pub trait NewRange {
    fn new(rangebounds: impl RangeBounds<usize>, length: usize) -> Self;
}

impl NewRange for Range<usize> {
    #[inline]
    #[track_caller]
    fn new(rangebounds: impl RangeBounds<usize>, length: usize) -> Self {
        #[cold]
        #[inline(never)]
        #[track_caller]
        fn assert_failed(start: usize, end: usize, len: usize) -> ! {
            if start > end {
                panic!("slice index starts at {start} but ends at {end}");
            } else {
                panic!("range end index {end} out of range for slice of length {len}");
            }
        }

        let start = match rangebounds.start_bound() {
            Bound::Included(&start) => start,
            Bound::Excluded(start) => {
                let Some(start) = start.checked_add(1) else {
                    panic!("attempted to index slice from after maximum usize")
                };
                start
            }
            Bound::Unbounded => 0
        };

        let end = match rangebounds.end_bound() {
            Bound::Included(end) => {
                let Some(end) = end.checked_add(1) else {
                    panic!("attempted to index slice up to maximum usize")
                };
                end
            }
            Bound::Excluded(&end) => end,
            Bound::Unbounded => length
        };

        if start > end || end > length {
            assert_failed(start, end, length);
        }

        Range {
            start,
            end
        }
    }
}
