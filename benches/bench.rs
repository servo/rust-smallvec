#![feature(test)]

#[macro_use]
extern crate smallvec;
extern crate test;

use smallvec::SmallVec;
use self::test::Bencher;

#[bench]
fn bench_push(b: &mut Bencher) {
    #[inline(never)]
    fn push_noinline(vec: &mut SmallVec<[u64; 16]>, x: u64) {
        vec.push(x)
    }

    b.iter(|| {
        let mut vec: SmallVec<[u64; 16]> = SmallVec::new();
        for x in 0..100 {
            push_noinline(&mut vec, x);
        }
        vec
    });
}

#[bench]
fn bench_insert(b: &mut Bencher) {
    #[inline(never)]
    fn insert_noinline(vec: &mut SmallVec<[u64; 16]>, x: u64) {
        vec.insert(0, x)
    }

    b.iter(|| {
        let mut vec: SmallVec<[u64; 16]> = SmallVec::new();
        for x in 0..100 {
            insert_noinline(&mut vec, x);
        }
        vec
    });
}

#[bench]
fn bench_insert_many(b: &mut Bencher) {
    #[inline(never)]
    fn insert_many_noinline<I: IntoIterator<Item=u64>>(
        vec: &mut SmallVec<[u64; 16]>, index: usize, iterable: I) {
        vec.insert_many(index, iterable)
    }

    b.iter(|| {
        let mut vec: SmallVec<[u64; 16]> = SmallVec::new();
        insert_many_noinline(&mut vec, 0, 0..100);
        insert_many_noinline(&mut vec, 0, 0..100);
        vec
    });
}

#[bench]
fn bench_extend(b: &mut Bencher) {
    b.iter(|| {
        let mut vec: SmallVec<[u64; 16]> = SmallVec::new();
        vec.extend(0..100);
        vec
    });
}

#[bench]
fn bench_from_slice(b: &mut Bencher) {
    let v: Vec<u64> = (0..100).collect();
    b.iter(|| {
        let vec: SmallVec<[u64; 16]> = SmallVec::from_slice(&v);
        vec
    });
}

#[bench]
fn bench_extend_from_slice(b: &mut Bencher) {
    let v: Vec<u64> = (0..100).collect();
    b.iter(|| {
        let mut vec: SmallVec<[u64; 16]> = SmallVec::new();
        vec.extend_from_slice(&v);
        vec
    });
}

#[bench]
fn bench_insert_from_slice(b: &mut Bencher) {
    let v: Vec<u64> = (0..100).collect();
    b.iter(|| {
        let mut vec: SmallVec<[u64; 16]> = SmallVec::new();
        vec.insert_from_slice(0, &v);
        vec.insert_from_slice(0, &v);
        vec
    });
}

#[bench]
fn bench_pushpop(b: &mut Bencher) {
    #[inline(never)]
    fn pushpop_noinline(vec: &mut SmallVec<[u64; 16]>, x: u64) {
        vec.push(x);
        vec.pop();
    }

    b.iter(|| {
        let mut vec: SmallVec<[u64; 16]> = SmallVec::new();
        for x in 0..100 {
            pushpop_noinline(&mut vec, x);
        }
        vec
    });
}

#[bench]
fn bench_macro_from_elem(b: &mut Bencher) {
    b.iter(|| {
        let vec: SmallVec<[u64; 16]> = smallvec![42; 100];
        vec
    });
}

#[bench]
fn bench_macro_from_list(b: &mut Bencher) {
    b.iter(|| {
        let vec: SmallVec<[u64; 16]> = smallvec![
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 20, 24, 32, 36, 0x40, 0x80,
            0x100, 0x200, 0x400, 0x800, 0x1000, 0x2000, 0x4000, 0x8000, 0x10000, 0x20000, 0x40000,
            0x80000, 0x100000
        ];
        vec
    });
}
#[bench]
fn bench_push_vec(b: &mut Bencher) {
    #[inline(never)]
    fn push_noinline(vec: &mut Vec<u64>, x: u64) {
        vec.push(x)
    }

    b.iter(|| {
        let mut vec: Vec<u64> = Vec::with_capacity(16);
        for x in 0..100 {
            push_noinline(&mut vec, x);
        }
        vec
    });
}

#[bench]
fn bench_insert_vec(b: &mut Bencher) {
    #[inline(never)]
    fn insert_noinline(vec: &mut Vec<u64>, x: u64) {
        vec.insert(0, x)
    }

    b.iter(|| {
        let mut vec: Vec<u64> = Vec::with_capacity(16);
        for x in 0..100 {
            insert_noinline(&mut vec, x);
        }
        vec
    });
}

#[bench]
fn bench_extend_vec(b: &mut Bencher) {
    b.iter(|| {
        let mut vec: Vec<u64> = Vec::with_capacity(16);
        vec.extend(0..100);
        vec
    });
}

#[bench]
fn bench_from_slice_vec(b: &mut Bencher) {
    let v: Vec<u64> = (0..100).collect();
    b.iter(|| {
        let vec: Vec<u64> = Vec::from(&v[..]);
        vec
    });
}

#[bench]
fn bench_extend_from_slice_vec(b: &mut Bencher) {
    let v: Vec<u64> = (0..100).collect();
    b.iter(|| {
        let mut vec: Vec<u64> = Vec::with_capacity(16);
        vec.extend_from_slice(&v);
        vec
    });
}

#[bench]
fn bench_pushpop_vec(b: &mut Bencher) {
    #[inline(never)]
    fn pushpop_noinline(vec: &mut Vec<u64>, x: u64) {
        vec.push(x);
        vec.pop();
    }

    b.iter(|| {
        let mut vec: Vec<u64> = Vec::with_capacity(16);
        for x in 0..100 {
            pushpop_noinline(&mut vec, x);
        }
        vec
    });
}

#[bench]
fn bench_macro_from_elem_vec(b: &mut Bencher) {
    b.iter(|| {
        let vec: Vec<u64> = vec![42; 100];
        vec
    });
}

#[bench]
fn bench_macro_from_list_vec(b: &mut Bencher) {
    b.iter(|| {
        let vec: Vec<u64> = vec![
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 20, 24, 32, 36, 0x40, 0x80,
            0x100, 0x200, 0x400, 0x800, 0x1000, 0x2000, 0x4000, 0x8000, 0x10000, 0x20000, 0x40000,
            0x80000, 0x100000
        ];
        vec
    });
}
