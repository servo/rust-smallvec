//! Simple fuzzer testing all available `SmallVec` operations
use smallvec::SmallVec;

// There's no point growing too much, so try not to grow
// over this size.
const CAP_GROWTH: usize = 256;

macro_rules! next_usize {
    ($b:ident) => {
        $b.next().unwrap_or(0) as usize
    };
}

macro_rules! next_u8 {
    ($b:ident) => {
        $b.next().unwrap_or(0)
    };
}

fn black_box_iter(i: impl Iterator<Item = u8>) {
    // print to work as a black_box
    print!("{}", i.fold(0u8, |acc, e| acc.wrapping_add(e)));
}

fn black_box_slice(s: &[u8]) {
    black_box_iter(s.iter().copied())
}

fn black_box_mut_slice(s: &mut [u8]) {
    s.iter_mut().map(|e| *e = e.wrapping_add(1)).count();
    black_box_iter((s as &[u8]).iter().copied())
}

fn do_test<A: smallvec::Array<Item = u8>>(data: &[u8]) -> SmallVec<A> {
    let mut v = SmallVec::<A>::new();

    let mut bytes = data.iter().copied();

    while let Some(op) = bytes.next() {
        match op % 27 {
            0 => {
                v = SmallVec::new();
            }
            1 => {
                v = SmallVec::with_capacity(next_usize!(bytes));
            }
            2 => {
                v = SmallVec::from_vec(v.to_vec());
            }
            3 => {
                black_box_iter(v.drain(..));
            }
            4 => {
                if v.len() < CAP_GROWTH {
                    v.push(next_u8!(bytes))
                }
            }
            5 => {
                v.pop();
            }
            6 => v.grow(next_usize!(bytes) + v.len()),
            7 => {
                if v.len() < CAP_GROWTH {
                    v.reserve(next_usize!(bytes))
                }
            }
            8 => {
                if v.len() < CAP_GROWTH {
                    v.reserve_exact(next_usize!(bytes))
                }
            }
            9 => v.shrink_to_fit(),
            10 => v.truncate(next_usize!(bytes)),
            11 => black_box_slice(v.as_slice()),
            12 => black_box_mut_slice(v.as_mut_slice()),
            13 => {
                if !v.is_empty() {
                    v.swap_remove(next_usize!(bytes) % v.len());
                }
            }
            14 => {
                v.clear();
            }
            15 => {
                if !v.is_empty() {
                    v.remove(next_usize!(bytes) % v.len());
                }
            }
            16 => {
                let insert_pos = next_usize!(bytes) % (v.len() + 1);
                v.insert(insert_pos, next_u8!(bytes));
            }
            17 => {
                let insert_pos = next_usize!(bytes) % (v.len() + 1);
                let how_many = next_usize!(bytes);
                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    v.insert_many(insert_pos, (0..how_many).map(|_| bytes.next().unwrap()));
                }));

                if result.is_err() {
                    assert!(bytes.next().is_none());
                }
            }
            18 => {
                v = SmallVec::from_vec(v.into_vec());
            }

            19 => {
                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    v.retain(|e| {
                        let alt_e = bytes.next().unwrap();
                        let retain = *e >= alt_e;
                        *e = e.wrapping_add(alt_e);
                        retain
                    });
                }));

                if result.is_err() {
                    assert!(bytes.next().is_none());
                }
            }
            20 => {
                v.dedup();
            }

            21 => {
                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    v.dedup_by(|a, b| {
                        let substitute = bytes.next().unwrap();
                        let dedup = a == b;
                        *a = a.wrapping_add(substitute);
                        *b = b.wrapping_add(substitute);
                        dedup
                    });
                }));

                if result.is_err() {
                    assert!(bytes.next().is_none());
                }
            }
            22 => {
                v = SmallVec::from_slice(data);
            }

            23 => {
                if v.len() < CAP_GROWTH {
                    v.extend_from_slice(data)
                }
            }

            24 => {
                if v.len() < CAP_GROWTH {
                    let insert_pos = next_usize!(bytes) % (v.len() + 1);
                    v.insert_from_slice(insert_pos, data);
                }
            }

            25 => {
                if v.len() < CAP_GROWTH {
                    v.resize(next_usize!(bytes), next_u8!(bytes));
                }
            }
            26 => {
                v = SmallVec::from_elem(next_u8!(bytes), next_usize!(bytes));
            }
            _ => panic!("booo"),
        }
    }
    v
}

fn do_test_all(data: &[u8]) {
    do_test::<[u8; 0]>(data);
    do_test::<[u8; 1]>(data);
    do_test::<[u8; 2]>(data);
    do_test::<[u8; 7]>(data);
    do_test::<[u8; 8]>(data);
}

#[cfg(feature = "afl")]
#[macro_use]
extern crate afl;
#[cfg(feature = "afl")]
fn main() {
    fuzz!(|data| {
        // Remove the panic hook so we can actually catch panic
        // See https://github.com/rust-fuzz/afl.rs/issues/150
        std::panic::set_hook(Box::new(|_| {}));
        do_test_all(data);
    });
}

#[cfg(feature = "honggfuzz")]
#[macro_use]
extern crate honggfuzz;
#[cfg(feature = "honggfuzz")]
fn main() {
    loop {
        fuzz!(|data| {
            // Remove the panic hook so we can actually catch panic
            // See https://github.com/rust-fuzz/afl.rs/issues/150
            std::panic::set_hook(Box::new(|_| {}));
            do_test_all(data);
        });
    }
}

#[cfg(test)]
mod tests {
    fn extend_vec_from_hex(hex: &str, out: &mut Vec<u8>) {
        let mut b = 0;
        for (idx, c) in hex.as_bytes().iter().enumerate() {
            b <<= 4;
            match *c {
                b'A'..=b'F' => b |= c - b'A' + 10,
                b'a'..=b'f' => b |= c - b'a' + 10,
                b'0'..=b'9' => b |= c - b'0',
                b'\n' => {}
                b' ' => {}
                _ => panic!("Bad hex"),
            }
            if (idx & 1) == 1 {
                out.push(b);
                b = 0;
            }
        }
    }

    #[test]
    fn duplicate_crash() {
        let mut a = Vec::new();
        // paste the output of `xxd -p <crash_dump>` here and run `cargo test`
        extend_vec_from_hex(
            r#"
            646e21f9f910f90200f9d9f9c7030000def9000010646e2af9f910f90264
            6e21f9f910f90200f9d9f9c7030000def90000106400f9f9d9f9c7030000
            def90000106400f9d9f9e7f1000000d9f9e7f1000000f9
            "#,
            &mut a,
        );
        super::do_test_all(&a);
    }
}
