use {
    iai_callgrind::{
        library_benchmark,
        library_benchmark_group,
        main
    },
    smallvec::SmallVec,
    std::hint::black_box
};

const INLINE_CAP: usize = 16;
const SPILLED_COUNT: usize = 100;

// =========================================================================
// PUSH
// =========================================================================

#[library_benchmark]
fn bench_smallvec_push_inline() -> SmallVec<u64, INLINE_CAP> {
    let mut v = SmallVec::<u64, INLINE_CAP>::new();
    for i in 0..(INLINE_CAP as u64) {
        v.push(black_box(i));
    }
    v
}

#[library_benchmark]
fn bench_smallvec_push_spilled() -> SmallVec<u64, INLINE_CAP> {
    let mut v = SmallVec::<u64, INLINE_CAP>::new();
    for i in 0..(SPILLED_COUNT as u64) {
        v.push(black_box(i));
    }
    v
}

#[library_benchmark]
fn bench_vec_push() -> Vec<u64> {
    let mut v = Vec::with_capacity(INLINE_CAP);
    for i in 0..(INLINE_CAP as u64) {
        v.push(black_box(i));
    }
    v
}

// =========================================================================
// POP
// =========================================================================

#[library_benchmark]
fn bench_smallvec_pop_inline() -> u64 {
    let mut v = SmallVec::<u64, INLINE_CAP>::new();
    for i in 0..(INLINE_CAP as u64) {
        v.push(black_box(i));
    }
    let mut sum = 0;
    while let Some(val) = v.pop() {
        sum += black_box(val);
    }
    sum
}

#[library_benchmark]
fn bench_smallvec_pop_spilled() -> u64 {
    let mut v = SmallVec::<u64, INLINE_CAP>::new();
    for i in 0..(SPILLED_COUNT as u64) {
        v.push(black_box(i));
    }
    let mut sum = 0;
    while let Some(val) = v.pop() {
        sum += black_box(val);
    }
    sum
}

// =========================================================================
// INSERT & REMOVE
// =========================================================================

#[library_benchmark]
fn bench_smallvec_insert_remove_inline() -> SmallVec<u64, INLINE_CAP> {
    let mut v = SmallVec::<u64, INLINE_CAP>::new();
    for i in 0..((INLINE_CAP - 1) as u64) {
        v.push(black_box(i));
    }
    v.insert(black_box(4), black_box(999));
    let _ = v.remove(black_box(4));
    v
}

#[library_benchmark]
fn bench_smallvec_insert_remove_spilled() -> SmallVec<u64, INLINE_CAP> {
    let mut v = SmallVec::<u64, INLINE_CAP>::new();
    for i in 0..(SPILLED_COUNT as u64) {
        v.push(black_box(i));
    }
    v.insert(black_box(50), black_box(999));
    let _ = v.remove(black_box(50));
    v
}

// =========================================================================
// EXTEND & FROM_SLICE
// =========================================================================

#[library_benchmark]
fn bench_smallvec_from_slice_inline() -> SmallVec<u64, INLINE_CAP> {
    let data = [42u64; INLINE_CAP];
    SmallVec::<u64, INLINE_CAP>::from(black_box(data.as_slice()))
}

#[library_benchmark]
fn bench_smallvec_from_slice_spilled() -> SmallVec<u64, INLINE_CAP> {
    let data = [42u64; SPILLED_COUNT];
    SmallVec::<u64, INLINE_CAP>::from(black_box(data.as_slice()))
}

#[library_benchmark]
fn bench_smallvec_extend_from_slice_inline() -> SmallVec<u64, INLINE_CAP> {
    let mut v = SmallVec::<u64, INLINE_CAP>::new();
    let data = [42u64; INLINE_CAP];
    v.extend_from_slice(black_box(&data));
    v
}

#[library_benchmark]
fn bench_smallvec_extend_from_slice_spilled() -> SmallVec<u64, INLINE_CAP> {
    let mut v = SmallVec::<u64, INLINE_CAP>::new();
    let data = [42u64; SPILLED_COUNT];
    v.extend_from_slice(black_box(&data));
    v
}

// =========================================================================
// ARRAY / BUF CONVERSION
// =========================================================================

#[library_benchmark]
fn bench_smallvec_from_array_inline() -> SmallVec<u64, INLINE_CAP> {
    SmallVec::<u64, INLINE_CAP>::from(black_box([1u64; INLINE_CAP]))
}

#[library_benchmark]
fn bench_smallvec_from_elem_spilled() -> SmallVec<u64, INLINE_CAP> {
    smallvec::from_elem(black_box(1u64), black_box(SPILLED_COUNT))
}

// =========================================================================
// ITERATION & DRAIN
// =========================================================================

#[library_benchmark]
fn bench_smallvec_into_iter_inline() -> u64 {
    let mut v = SmallVec::<u64, INLINE_CAP>::new();
    for i in 0..(INLINE_CAP as u64) {
        v.push(black_box(i));
    }
    v.into_iter().fold(0, |acc, x| acc + black_box(x))
}

#[library_benchmark]
fn bench_smallvec_into_iter_spilled() -> u64 {
    let mut v = SmallVec::<u64, INLINE_CAP>::new();
    for i in 0..(SPILLED_COUNT as u64) {
        v.push(black_box(i));
    }
    v.into_iter().fold(0, |acc, x| acc + black_box(x))
}

#[library_benchmark]
fn bench_smallvec_drain_inline() -> u64 {
    let mut v = SmallVec::<u64, INLINE_CAP>::new();
    for i in 0..(INLINE_CAP as u64) {
        v.push(black_box(i));
    }
    let sum = v.drain(..).fold(0, |acc, x| acc + black_box(x));
    black_box(v);
    sum
}

#[library_benchmark]
fn bench_smallvec_drain_spilled() -> u64 {
    let mut v = SmallVec::<u64, INLINE_CAP>::new();
    for i in 0..(SPILLED_COUNT as u64) {
        v.push(black_box(i));
    }
    let sum = v.drain(..).fold(0, |acc, x| acc + black_box(x));
    black_box(v);
    sum
}

// =========================================================================
// RETAIN MUT
// =========================================================================

#[library_benchmark]
fn bench_smallvec_retain_mut_inline() -> SmallVec<u64, INLINE_CAP> {
    let mut v = SmallVec::<u64, INLINE_CAP>::new();
    for i in 0..(INLINE_CAP as u64) {
        v.push(black_box(i));
    }
    v.retain_mut(|x| *x % 2 == 0);
    v
}

#[library_benchmark]
fn bench_smallvec_retain_mut_spilled() -> SmallVec<u64, INLINE_CAP> {
    let mut v = SmallVec::<u64, INLINE_CAP>::new();
    for i in 0..(SPILLED_COUNT as u64) {
        v.push(black_box(i));
    }
    v.retain_mut(|x| *x % 2 == 0);
    v
}

// =========================================================================
// GROUPS & MAIN
// =========================================================================

library_benchmark_group!(
    name = push_group;
    benchmarks =
        bench_smallvec_push_inline,
        bench_smallvec_push_spilled,
        bench_vec_push,
);

library_benchmark_group!(
    name = pop_group;
    benchmarks =
        bench_smallvec_pop_inline,
        bench_smallvec_pop_spilled,
);

library_benchmark_group!(
    name = insert_remove_group;
    benchmarks =
        bench_smallvec_insert_remove_inline,
        bench_smallvec_insert_remove_spilled,
);

library_benchmark_group!(
    name = slice_group;
    benchmarks =
        bench_smallvec_from_slice_inline,
        bench_smallvec_from_slice_spilled,
        bench_smallvec_extend_from_slice_inline,
        bench_smallvec_extend_from_slice_spilled,
);

library_benchmark_group!(
    name = array_group;
    benchmarks =
        bench_smallvec_from_array_inline,
        bench_smallvec_from_elem_spilled,
);

library_benchmark_group!(
    name = iteration_group;
    benchmarks =
        bench_smallvec_into_iter_inline,
        bench_smallvec_into_iter_spilled,
        bench_smallvec_drain_inline,
        bench_smallvec_drain_spilled,
        bench_smallvec_retain_mut_inline,
        bench_smallvec_retain_mut_spilled,
);

main!(
    library_benchmark_groups = push_group,
    pop_group,
    insert_remove_group,
    slice_group,
    array_group,
    iteration_group
);
