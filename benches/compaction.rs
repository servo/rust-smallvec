#![feature(test)]
extern crate test;
use smallvec::SmallVec;
use test::{black_box, Bencher};
fn bench(b: &mut Bencher, n: usize, mode: usize, wide: bool, retain: bool) {
    if wide {
        run::<[u64; 16]>(b, n, mode, retain)
    } else {
        run::<[u64; 1]>(b, n, mode, retain)
    }
}
trait Payload: Copy {
    fn from_key(key: u64) -> Self;
    fn key(&self) -> u64;
}
impl Payload for [u64; 1] {
    fn from_key(key: u64) -> Self {
        [key; 1]
    }
    fn key(&self) -> u64 {
        self[0]
    }
}
impl Payload for [u64; 16] {
    fn from_key(key: u64) -> Self {
        [key; 16]
    }
    fn key(&self) -> u64 {
        self[0]
    }
}
fn run<T: Payload>(b: &mut Bencher, n: usize, mode: usize, retain: bool) {
    let input: Vec<_> = (0..n)
        .map(|i| {
            T::from_key(if retain || mode == 0 {
                i as u64
            } else if mode == 1 {
                (i / 2) as u64
            } else {
                0
            })
        })
        .collect();
    let mut values = SmallVec::<[T; 16]>::from_slice(&input);
    // libtest includes restoring the input; both revisions use the same setup.
    b.iter(|| {
        values.clear();
        values.extend_from_slice(black_box(&input));
        if retain {
            values.retain(|x| match mode {
                0 => true,
                1 => x.key() % 2 == 0,
                _ => false,
            });
        } else {
            values.dedup_by(|a, b| a.key() == b.key());
        }
        black_box(values.as_slice());
    });
}
macro_rules! case {
    ($name:ident,$n:expr,$mode:expr,$wide:expr,$retain:expr) => {
        #[bench]
        fn $name(b: &mut Bencher) {
            bench(b, $n, $mode, $wide, $retain)
        }
    };
}
case!(dedup_all_kept_16, 16, 0, false, false);
case!(dedup_half_kept_16, 16, 1, false, false);
case!(dedup_one_or_none_16, 16, 2, false, false);
case!(dedup_all_kept_17, 17, 0, false, false);
case!(dedup_half_kept_17, 17, 1, false, false);
case!(dedup_one_or_none_17, 17, 2, false, false);
case!(dedup_all_kept_4096, 4096, 0, true, false);
case!(dedup_half_kept_4096, 4096, 1, true, false);
case!(dedup_one_or_none_4096, 4096, 2, true, false);
case!(retain_all_kept_16, 16, 0, false, true);
case!(retain_half_kept_16, 16, 1, false, true);
case!(retain_one_or_none_16, 16, 2, false, true);
case!(retain_all_kept_17, 17, 0, false, true);
case!(retain_half_kept_17, 17, 1, false, true);
case!(retain_one_or_none_17, 17, 2, false, true);
case!(retain_all_kept_4096, 4096, 0, true, true);
case!(retain_half_kept_4096, 4096, 1, true, true);
case!(retain_one_or_none_4096, 4096, 2, true, true);
case!(dedup_u64_half_4096, 4096, 1, false, false);
case!(retain_u64_half_4096, 4096, 1, false, true);

#[bench]
fn vec_dedup_pairs_4096_control(b: &mut Bencher) {
    let input: Vec<_> = (0..4096).map(|i| [(i / 2) as u64; 16]).collect();
    let mut values = input.clone();
    b.iter(|| {
        values.clear();
        values.extend_from_slice(black_box(&input));
        values.dedup_by(|a, b| a[0] == b[0]);
        black_box(values.as_slice());
    });
}
