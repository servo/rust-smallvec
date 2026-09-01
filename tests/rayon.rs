use {
    rayon::{
        iter::ParallelIterator,
        prelude::ParallelSlice
    },
    smallvec::SmallVec
};

#[test]
fn rayon() {
    assert_eq!(
        [0, 1, 2, 3]
            .par_chunks(2)
            .map(SmallVec::<i32, 2>::from)
            .flatten()
            .collect::<Vec<i32>>(),
        [0, 1, 2, 3]
    );
}
