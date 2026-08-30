#![allow(deprecated)]

use {
    criterion::{
        Bencher,
        Criterion,
        criterion_group,
        criterion_main
    },
    smallvec::{
        SmallVec,
        smallvec
    },
    std::{
        hint::black_box,
        time::Duration
    }
};

const VEC_SIZE: usize = 16;
const SPILLED_SIZE: usize = 100;

trait Vector<T>: for<'a> From<&'a [T]> + Extend<T> {
    fn new() -> Self;
    fn push(&mut self, val: T);
    fn pop(&mut self) -> Option<T>;
    fn remove(&mut self, p: usize) -> T;
    fn insert(&mut self, n: usize, val: T);
    fn from_elem(val: T, n: usize) -> Self;
    fn from_elems(val: &[T]) -> Self;
    fn extend_from_slice(&mut self, other: &[T]);
    fn retain_mut<F>(&mut self, f: F)
    where F: FnMut(&mut T) -> bool;
}

impl<T: Copy> Vector<T> for Vec<T> {
    fn new() -> Self { Self::with_capacity(VEC_SIZE) }

    fn push(&mut self, val: T) { self.push(val) }

    fn pop(&mut self) -> Option<T> { self.pop() }

    fn remove(&mut self, p: usize) -> T { self.remove(p) }

    fn insert(&mut self, n: usize, val: T) { self.insert(n, val) }

    fn from_elem(val: T, n: usize) -> Self { vec![val; n] }

    fn from_elems(val: &[T]) -> Self { val.to_owned() }

    fn extend_from_slice(&mut self, other: &[T]) { Vec::extend_from_slice(self, other) }

    fn retain_mut<F>(&mut self, f: F)
    where F: FnMut(&mut T) -> bool {
        self.retain_mut(f)
    }
}

impl<T: Copy> Vector<T> for SmallVec<T, VEC_SIZE> {
    fn new() -> Self { Self::new() }

    fn push(&mut self, val: T) { self.push(val) }

    fn pop(&mut self) -> Option<T> { self.pop() }

    fn remove(&mut self, p: usize) -> T { self.remove(p) }

    fn insert(&mut self, n: usize, val: T) { self.insert(n, val) }

    fn from_elem(val: T, n: usize) -> Self { smallvec![val; n] }

    fn from_elems(val: &[T]) -> Self { SmallVec::from(val) }

    fn extend_from_slice(&mut self, other: &[T]) { SmallVec::extend_from_slice(self, other) }

    fn retain_mut<F>(&mut self, f: F)
    where F: FnMut(&mut T) -> bool {
        self.retain_mut(f)
    }
}

macro_rules! make_benches {
    ($typ:ty { $($b_name:ident => $g_name:ident($($args:expr),*),)* }) => {
        $(
            fn $b_name(c: &mut Criterion) {
                c.bench_function(stringify!($b_name), |b: &mut Bencher| {
                    $g_name::<$typ>($($args,)* b)
                });
            }
        )*
    }
}

// ----------  Bench generation (same list, just using the new macro)
// ----------
make_benches! {
    SmallVec<u64, VEC_SIZE> {
        bench_push => gen_push(SPILLED_SIZE as _),
        bench_push_small => gen_push(VEC_SIZE as _),
        bench_insert_push => gen_insert_push(SPILLED_SIZE as _),
        bench_insert_push_small => gen_insert_push(VEC_SIZE as _),
        bench_insert => gen_insert(SPILLED_SIZE as _),
        bench_insert_small => gen_insert(VEC_SIZE as _),
        bench_remove => gen_remove(SPILLED_SIZE as _),
        bench_remove_small => gen_remove(VEC_SIZE as _),
        bench_extend => gen_extend(SPILLED_SIZE as _),
        bench_extend_small => gen_extend(VEC_SIZE as _),
        bench_extend_filtered => gen_extend_filtered(SPILLED_SIZE as _),
        bench_extend_filtered_small => gen_extend_filtered(VEC_SIZE as _),
        bench_from_iter => gen_from_iter(SPILLED_SIZE as _),
        bench_from_iter_small => gen_from_iter(VEC_SIZE as _),
        bench_from_slice => gen_from_slice(SPILLED_SIZE as _),
        bench_from_slice_small => gen_from_slice(VEC_SIZE as _),
        bench_extend_from_slice => gen_extend_from_slice(SPILLED_SIZE as _),
        bench_extend_from_slice_small => gen_extend_from_slice(VEC_SIZE as _),
        bench_macro_from_elem => gen_from_elem(SPILLED_SIZE as _),
        bench_macro_from_elem_small => gen_from_elem(VEC_SIZE as _),
        bench_pushpop => gen_pushpop(),
        bench_retain_mut_half => gen_retain_mut_half(SPILLED_SIZE as _),
        bench_retain_mut_half_small => gen_retain_mut_half(VEC_SIZE as _),
        bench_retain_mut_all => gen_retain_mut_all(SPILLED_SIZE as _),
        bench_retain_mut_all_small => gen_retain_mut_all(VEC_SIZE as _),
        bench_retain_mut_none => gen_retain_mut_none(SPILLED_SIZE as _),
        bench_retain_mut_none_small => gen_retain_mut_none(VEC_SIZE as _),
    }
}

make_benches! {
    Vec<u64> {
        bench_push_vec => gen_push(SPILLED_SIZE as _),
        bench_push_vec_small => gen_push(VEC_SIZE as _),
        bench_insert_push_vec => gen_insert_push(SPILLED_SIZE as _),
        bench_insert_push_vec_small => gen_insert_push(VEC_SIZE as _),
        bench_insert_vec => gen_insert(SPILLED_SIZE as _),
        bench_insert_vec_small => gen_insert(VEC_SIZE as _),
        bench_remove_vec => gen_remove(SPILLED_SIZE as _),
        bench_remove_vec_small => gen_remove(VEC_SIZE as _),
        bench_extend_vec => gen_extend(SPILLED_SIZE as _),
        bench_extend_vec_small => gen_extend(VEC_SIZE as _),
        bench_extend_vec_filtered => gen_extend_filtered(SPILLED_SIZE as _),
        bench_extend_vec_filtered_small => gen_extend_filtered(VEC_SIZE as _),
        bench_from_iter_vec => gen_from_iter(SPILLED_SIZE as _),
        bench_from_iter_vec_small => gen_from_iter(VEC_SIZE as _),
        bench_from_slice_vec => gen_from_slice(SPILLED_SIZE as _),
        bench_from_slice_vec_small => gen_from_slice(VEC_SIZE as _),
        bench_extend_from_slice_vec => gen_extend_from_slice(SPILLED_SIZE as _),
        bench_extend_from_slice_vec_small => gen_extend_from_slice(VEC_SIZE as _),
        bench_macro_from_elem_vec => gen_from_elem(SPILLED_SIZE as _),
        bench_macro_from_elem_vec_small => gen_from_elem(VEC_SIZE as _),
        bench_pushpop_vec => gen_pushpop(),
        bench_retain_mut_vec_half => gen_retain_mut_half(SPILLED_SIZE as _),
        bench_retain_mut_vec_half_small => gen_retain_mut_half(VEC_SIZE as _),
        bench_retain_mut_vec_all => gen_retain_mut_all(SPILLED_SIZE as _),
        bench_retain_mut_vec_all_small => gen_retain_mut_all(VEC_SIZE as _),
        bench_retain_mut_vec_none => gen_retain_mut_none(SPILLED_SIZE as _),
        bench_retain_mut_vec_none_small => gen_retain_mut_none(VEC_SIZE as _),
    }
}

fn gen_push<V: Vector<u64>>(n: u64, b: &mut Bencher) {
    #[inline(never)]
    fn push_noinline<V: Vector<u64>>(vec: &mut V, x: u64) { vec.push(black_box(x)); }

    b.iter(|| {
        let n = black_box(n);
        let mut vec = V::new();
        for x in 0..n {
            push_noinline(&mut vec, x);
        }
        black_box(vec)
    });
}

fn gen_insert_push<V: Vector<u64>>(n: u64, b: &mut Bencher) {
    #[inline(never)]
    fn insert_push_noinline<V: Vector<u64>>(vec: &mut V, x: u64) {
        vec.insert(black_box(x) as usize, black_box(x));
    }

    b.iter(|| {
        let n = black_box(n);
        let mut vec = V::new();
        for x in 0..n {
            insert_push_noinline(&mut vec, x);
        }
        black_box(vec)
    });
}

fn gen_insert<V: Vector<u64>>(n: u64, b: &mut Bencher) {
    #[inline(never)]
    fn insert_noinline<V: Vector<u64>>(vec: &mut V, p: usize, x: u64) {
        vec.insert(black_box(p), black_box(x))
    }

    b.iter_with_setup(
        || {
            let mut vec = V::new();
            vec.push(0);
            vec
        },
        |mut vec| {
            let n = black_box(n);
            for x in 0..n {
                insert_noinline(&mut vec, 0, x);
            }
            vec
        }
    );
}

fn gen_remove<V: Vector<u64>>(n: usize, b: &mut Bencher) {
    #[inline(never)]
    fn remove_noinline<V: Vector<u64>>(vec: &mut V, p: usize) -> u64 { vec.remove(black_box(p)) }

    b.iter_with_setup(
        || V::from_elem(0, black_box(n)),
        |mut vec| {
            for _ in 0..n {
                black_box(remove_noinline(&mut vec, 0));
            }
            vec
        }
    );
}

fn gen_extend<V: Vector<u64>>(n: u64, b: &mut Bencher) {
    b.iter(|| {
        let n = black_box(n);
        let mut vec = V::new();
        vec.extend(0..n);
        black_box(vec)
    });
}

fn gen_extend_filtered<V: Vector<u64>>(n: u64, b: &mut Bencher) {
    b.iter(|| {
        let mut vec = V::new();
        vec.extend((0..black_box(n)).filter(|i| black_box(*i) % 2 == 0));
        black_box(vec)
    });
}

fn gen_from_iter<V: Vector<u64>>(n: u64, b: &mut Bencher) {
    let v: Vec<u64> = (0..black_box(n)).collect();
    b.iter(|| {
        let vec = V::from(black_box(&v));
        black_box(vec)
    });
}

fn gen_from_slice<V: Vector<u64>>(n: u64, b: &mut Bencher) {
    let v: Vec<u64> = (0..black_box(n)).collect();
    b.iter(|| {
        let vec = V::from_elems(black_box(&v));
        black_box(vec)
    });
}

fn gen_extend_from_slice<V: Vector<u64>>(n: u64, b: &mut Bencher) {
    let v: Vec<u64> = (0..black_box(n)).collect();
    b.iter(|| {
        let mut vec = V::new();
        vec.extend_from_slice(black_box(&v));
        black_box(vec)
    });
}

fn gen_pushpop<V: Vector<u64>>(b: &mut Bencher) {
    #[inline(never)]
    fn pushpop_noinline<V: Vector<u64>>(vec: &mut V, x: u64) -> Option<u64> {
        vec.push(black_box(x));
        vec.pop()
    }

    b.iter(|| {
        let mut vec = V::new();
        for x in 0..SPILLED_SIZE as _ {
            black_box(pushpop_noinline(&mut vec, x));
        }
        black_box(vec)
    });
}

fn gen_from_elem<V: Vector<u64>>(n: usize, b: &mut Bencher) {
    b.iter(|| {
        let n = black_box(n);
        let vec = V::from_elem(black_box(42), n);
        black_box(vec)
    });
}

fn gen_retain_mut_half<V: Vector<u64>>(n: usize, b: &mut Bencher) {
    b.iter_with_setup(
        || V::from_elem(16, black_box(n)),
        |mut vec| {
            vec.retain_mut(|x| black_box(*x) % 2 == 0);
            vec
        }
    );
}

fn gen_retain_mut_all<V: Vector<u64>>(n: usize, b: &mut Bencher) {
    b.iter_with_setup(
        || V::from_elem(16, black_box(n)),
        |mut vec| {
            vec.retain_mut(|_| true);
            vec
        }
    );
}

fn gen_retain_mut_none<V: Vector<u64>>(n: usize, b: &mut Bencher) {
    b.iter_with_setup(
        || V::from_elem(16, black_box(n)),
        |mut vec| {
            vec.retain_mut(|_| false);
            vec
        }
    );
}

fn bench_macro_from_list(c: &mut Criterion) {
    c.bench_function("bench_macro_from_list", |b| {
        b.iter(|| {
            let vec: SmallVec<u64, 16> = smallvec![
                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 20, 24, 32, 36, 0x40,
                0x80, 0x100, 0x200, 0x400, 0x800, 0x1000, 0x2000, 0x4000, 0x8000, 0x10000, 0x20000,
                0x40000, 0x80000, 0x100000,
            ];
            vec
        })
    });
}

fn bench_macro_from_list_vec(c: &mut Criterion) {
    c.bench_function("bench_macro_from_list_vec", |b| {
        b.iter(|| {
            let vec: Vec<u64> = vec![
                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 20, 24, 32, 36, 0x40,
                0x80, 0x100, 0x200, 0x400, 0x800, 0x1000, 0x2000, 0x4000, 0x8000, 0x10000, 0x20000,
                0x40000, 0x80000, 0x100000,
            ];
            vec
        })
    });
}

criterion_group!(
    name = benches;
    config = Criterion::default()
        .warm_up_time(Duration::from_millis(200))
        .measurement_time(Duration::from_millis(700));
    targets =
    bench_push,
    bench_push_small,
    bench_insert_push,
    bench_insert_push_small,
    bench_insert,
    bench_insert_small,
    bench_remove,
    bench_remove_small,
    bench_extend,
    bench_extend_small,
    bench_extend_filtered,
    bench_extend_filtered_small,
    bench_from_iter,
    bench_from_iter_small,
    bench_from_slice,
    bench_from_slice_small,
    bench_extend_from_slice,
    bench_extend_from_slice_small,
    bench_macro_from_elem,
    bench_macro_from_elem_small,
    bench_pushpop,
    bench_retain_mut_half,
    bench_retain_mut_half_small,
    bench_retain_mut_all,
    bench_retain_mut_all_small,
    bench_retain_mut_none,
    bench_retain_mut_none_small,
    bench_push_vec,
    bench_push_vec_small,
    bench_insert_push_vec,
    bench_insert_push_vec_small,
    bench_insert_vec,
    bench_insert_vec_small,
    bench_remove_vec,
    bench_remove_vec_small,
    bench_extend_vec,
    bench_extend_vec_small,
    bench_extend_vec_filtered,
    bench_extend_vec_filtered_small,
    bench_from_iter_vec,
    bench_from_iter_vec_small,
    bench_from_slice_vec,
    bench_from_slice_vec_small,
    bench_extend_from_slice_vec,
    bench_extend_from_slice_vec_small,
    bench_macro_from_elem_vec,
    bench_macro_from_elem_vec_small,
    bench_pushpop_vec,
    bench_retain_mut_vec_half,
    bench_retain_mut_vec_half_small,
    bench_retain_mut_vec_all,
    bench_retain_mut_vec_all_small,
    bench_retain_mut_vec_none,
    bench_retain_mut_vec_none_small,
    bench_macro_from_list,
    bench_macro_from_list_vec
);
criterion_main!(benches);
