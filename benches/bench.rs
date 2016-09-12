#![feature(test)]

extern crate smallvec;
extern crate test;

use smallvec::SmallVec;
use self::test::Bencher;

#[bench]
fn bench_push(b: &mut Bencher) {
    b.iter(|| {
        let mut vec: SmallVec<[u64; 16]> = SmallVec::new();
        for x in 0..100 {
            vec.push(x);
        }
        vec
    });
}

#[bench]
fn bench_insert(b: &mut Bencher) {
    b.iter(|| {
        let mut vec: SmallVec<[u64; 16]> = SmallVec::new();
        for x in 0..100 {
            vec.insert(0, x);
        }
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
fn bench_pushpop(b: &mut Bencher) {
    b.iter(|| {
        let mut vec: SmallVec<[u64; 16]> = SmallVec::new();
        for x in 0..100 {
            vec.push(x);
            vec.pop();
        }
        vec
    });
}
