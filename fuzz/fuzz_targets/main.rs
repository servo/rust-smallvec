//! Simple fuzzer testing all available `SmallVec` operations

#![no_main]

use {arbitrary::Arbitrary, libfuzzer_sys::fuzz_target, smallvec::SmallVec, std::fmt::Debug};

/// A generic wrapper that bounds data generated via `arbitrary`.
/// Default cap is 255.
#[derive(Debug, Clone)]
pub struct Bounded<T, const CAP: usize = 255>(pub T);

// Bounded `usize` between `0..=CAP`
impl<'a, const CAP: usize> Arbitrary<'a> for Bounded<usize, CAP> {
    #[inline]
    fn arbitrary(u: &mut arbitrary::Unstructured<'a>) -> arbitrary::Result<Self> {
        Ok(Bounded(u.int_in_range(0..=CAP)?))
    }
}

// Bounded `Vec<T>` whose length is between `0..=CAP`
impl<'a, T, const CAP: usize> Arbitrary<'a> for Bounded<Vec<T>, CAP>
where
    T: Arbitrary<'a>,
{
    fn arbitrary(u: &mut arbitrary::Unstructured<'a>) -> arbitrary::Result<Self> {
        let len = u.int_in_range(0..=CAP)?;
        let vec = u
            .arbitrary_iter()?
            .take(len)
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Bounded(vec))
    }
}

#[inline]
fn choose_range(
    u: &mut arbitrary::Unstructured,
    len: usize,
) -> arbitrary::Result<std::ops::Range<usize>> {
    let start = u.int_in_range(0..=len)?;
    let end = u.int_in_range(start..=len)?;
    Ok(start..end)
}

#[derive(Arbitrary, Debug)]
enum Op {
    New,
    WithCapacity(Bounded<usize>),
    FromVec,
    FromSlice(Bounded<Vec<usize>>),
    Push(usize),
    Pop,
    Grow(Bounded<usize>),
    Reserve(Bounded<usize>),
    ReserveExact(Bounded<usize>),
    ShrinkToFit,
    Truncate,
    SwapRemove,
    Clear,
    Remove,
    Insert(usize),
    Drain,
    RetainEven,
    Dedup,
    ExtendFromSlice(Bounded<Vec<usize>>),
    Resize(Bounded<usize>, usize),
}

/// Helper to assert equivalence of all structural invariants of `SmallVec`
/// against `alloc::Vec`
fn assert_invariants<T: Copy + PartialEq + Debug, const N: usize>(
    small_vec: &mut SmallVec<[T; N]>,
    std_vec: &mut Vec<T>,
) {
    // Length and content equivalence
    assert_eq!(small_vec.len(), std_vec.len(), "`len()` mismatch");
    assert_eq!(
        small_vec.is_empty(),
        std_vec.is_empty(),
        "`is_empty()` mismatch"
    );
    assert_eq!(
        small_vec.as_slice(),
        std_vec.as_slice(),
        "`as_slice()` mismatch"
    );

    // Capacity & spilling invariants
    assert!(
        small_vec.capacity() >= small_vec.len(),
        "`capacity()` is smaller than `len()`"
    );
    assert!(
        small_vec.capacity() >= N,
        "`capacity()` is smaller than inline size `N`"
    );
    assert_eq!(
        small_vec.spilled(),
        small_vec.capacity() > N,
        "`spilled()` doesn't equal to `capacity() > N`"
    );

    // Iterator invariants
    assert!(
        small_vec.iter().eq(std_vec.iter()),
        "iterator yield mismatch"
    );
    assert!(
        small_vec.iter().rev().eq(std_vec.iter().rev()),
        "reverse iterator yield mismatch"
    );
    assert!(
        small_vec
            .clone()
            .into_iter()
            .eq(std_vec.clone().into_iter()),
        "`into_iter()` yield mismatch"
    );
    assert_eq!(
        small_vec.iter().size_hint(),
        std_vec.iter().size_hint(),
        "`size_hint()` mismatch"
    );
    assert_eq!(
        small_vec.iter().len(),
        std_vec.iter().len(),
        "`ExactSizeIterator::len()` mismatch"
    );

    // Clone invariant
    assert_eq!(&small_vec.clone()[..], std_vec, "clone mismatch");
}

fn test_with_inline_cap<const N: usize>(
    u: &mut arbitrary::Unstructured,
    ops: &[Op],
) -> arbitrary::Result<()> {
    // We let `T` be `usize` instead of `u8` because, albeit less efficient,
    // this incurs potential memory misalignment which should be properly
    // handled by the library.

    let mut small_vec = SmallVec::<[usize; N]>::new();
    let mut std_vec = Vec::<usize>::new();

    for op in ops {
        match op {
            Op::New => {
                small_vec = SmallVec::new();
                std_vec = Vec::new();
            }
            Op::WithCapacity(cap) => {
                small_vec = SmallVec::with_capacity(cap.0);
                std_vec = Vec::with_capacity(cap.0);
            }
            Op::FromVec => {
                small_vec = SmallVec::from_vec(small_vec.into_vec());
                // No-op on `Vec`
            }
            Op::FromSlice(data) => {
                small_vec = SmallVec::from(data.0.as_slice());
                std_vec = data.0.clone();
            }
            Op::Push(val) => {
                small_vec.push(*val);
                std_vec.push(*val);
            }
            Op::Pop => {
                assert_eq!(small_vec.pop(), std_vec.pop(), "`pop()` mismatch");
            }
            Op::Grow(target) => {
                small_vec.grow(target.0);
                // Mimic `SmallVec::grow` on `Vec`
                if target.0 > std_vec.capacity() {
                    let additional = target.0 - std_vec.len();
                    std_vec.reserve(additional);
                }
            }
            Op::Reserve(amount) => {
                small_vec.reserve(amount.0);
                std_vec.reserve(amount.0);
            }
            Op::ReserveExact(amount) => {
                small_vec.reserve_exact(amount.0);
                std_vec.reserve_exact(amount.0);
            }
            Op::ShrinkToFit => {
                small_vec.shrink_to_fit();
                std_vec.shrink_to_fit();
            }
            Op::Truncate => {
                let len = u.int_in_range(0..=small_vec.len())?;
                small_vec.truncate(len);
                std_vec.truncate(len);
            }
            Op::SwapRemove => {
                if !small_vec.is_empty() {
                    let idx = u.choose_index(small_vec.len())?;
                    assert_eq!(
                        small_vec.swap_remove(idx),
                        std_vec.swap_remove(idx),
                        "`swap_remove()` mismatch"
                    );
                }
            }
            Op::Clear => {
                small_vec.clear();
                std_vec.clear();
            }
            Op::Remove => {
                if !small_vec.is_empty() {
                    let idx = u.choose_index(small_vec.len())?;
                    assert_eq!(
                        small_vec.remove(idx),
                        std_vec.remove(idx),
                        "`remove()` mismatch"
                    );
                }
            }
            Op::Insert(val) => {
                let idx = u.int_in_range(0..=small_vec.len())?;
                small_vec.insert(idx, *val);
                std_vec.insert(idx, *val);
            }
            Op::Drain => {
                let len = small_vec.len();
                let range = choose_range(u, len)?;

                let small_vec_drained = small_vec.drain(range.clone());
                let std_vec_drained = std_vec.drain(range);

                assert!(
                    small_vec_drained.eq(std_vec_drained),
                    "`drain()` yield mismatch"
                );
            }
            Op::RetainEven => {
                small_vec.retain(|&mut e| e % 2 == 0);
                std_vec.retain(|e| e % 2 == 0);
            }
            Op::Dedup => {
                small_vec.dedup();
                std_vec.dedup();
            }
            Op::ExtendFromSlice(items) => {
                small_vec.extend_from_slice(&items.0);
                std_vec.extend_from_slice(&items.0);
            }
            Op::Resize(new_len, val) => {
                small_vec.resize(new_len.0, *val);
                std_vec.resize(new_len.0, *val);
            }
        }

        assert_invariants(&mut small_vec, &mut std_vec);
    }

    Ok(())
}

fn run_test(mut u: arbitrary::Unstructured) -> arbitrary::Result<()> {
    let ops = Vec::<Op>::arbitrary(&mut u)?;
    let dynamic_entropy = u.take_rest();

    let run_test = |test_func: fn(&mut arbitrary::Unstructured, &[Op]) -> arbitrary::Result<()>| {
        test_func(&mut arbitrary::Unstructured::new(dynamic_entropy), &ops)
    };

    run_test(test_with_inline_cap::<0>)?;
    run_test(test_with_inline_cap::<1>)?;
    run_test(test_with_inline_cap::<2>)?;
    run_test(test_with_inline_cap::<7>)?;
    run_test(test_with_inline_cap::<8>)?;

    Ok(())
}

fuzz_target!(|data: &[u8]| {
    let u = arbitrary::Unstructured::new(data);

    let _ = run_test(u);
});
