use smallvec::SmallVec;
use std::{
    cell::Cell,
    panic::{catch_unwind, AssertUnwindSafe},
    rc::Rc,
};
type V<T> = SmallVec<[T; 16]>;

struct Tracked {
    id: usize,
    drops: Rc<Vec<Cell<usize>>>,
    panic_at: Option<usize>,
}
impl Drop for Tracked {
    fn drop(&mut self) {
        self.drops[self.id].set(self.drops[self.id].get() + 1);
        assert_ne!(self.panic_at, Some(self.id), "drop panic");
    }
}
fn tracked(len: usize, panic_at: Option<usize>) -> (Vec<Tracked>, Rc<Vec<Cell<usize>>>) {
    let drops = Rc::new((0..len).map(|_| Cell::new(0)).collect::<Vec<_>>());
    let values = (0..len)
        .map(|id| Tracked {
            id,
            drops: drops.clone(),
            panic_at,
        })
        .collect();
    (values, drops)
}
fn ids(values: &[Tracked]) -> Vec<usize> {
    values.iter().map(|v| v.id).collect()
}

thread_local! { static ZST_DROPS: Cell<usize> = Cell::new(0); }
struct Zst;
impl Drop for Zst {
    fn drop(&mut self) {
        ZST_DROPS.with(|x| x.set(x.get() + 1));
    }
}
#[test]
fn retain_panic_preserves_unprocessed_tail() {
    for len in [8, 32].iter().copied() {
        for panic_at in 0..len {
            for drop_panics in [false, true].iter().copied() {
                if drop_panics && panic_at % 2 == 0 {
                    continue;
                }
                let destructor = if drop_panics { Some(panic_at) } else { None };
                let (input, drops) = tracked(len, destructor);
                let mut actual: V<_> = input.into_iter().collect();
                assert!(catch_unwind(AssertUnwindSafe(|| actual.retain(|x| {
                    if !drop_panics {
                        assert_ne!(x.id, panic_at);
                    }
                    x.id % 2 == 0
                })))
                .is_err());
                let read = panic_at + if drop_panics { 1 } else { 0 };
                let expected: Vec<_> = (0..panic_at).step_by(2).chain(read..len).collect();
                assert_eq!(ids(&actual), expected);
                drop(actual);
                assert!(drops.iter().all(|x| x.get() == 1));
            }
        }
    }
}
#[test]
fn retain_patterns_and_zst() {
    for len in [0, 1, 15, 16, 17, 64].iter().copied() {
        for keep in 0..3 {
            let mut actual: V<_> = (0..len).collect();
            let mut expected: Vec<_> = (0..len).collect();
            actual.retain(|x| {
                *x += 1;
                *x % 2 < keep
            });
            for x in &mut expected {
                *x += 1;
            }
            expected.retain(|x| *x % 2 < keep);
            assert_eq!(actual.as_slice(), expected.as_slice());
        }
    }
    ZST_DROPS.with(|x| x.set(0));
    let mut values: V<_> = (0..32).map(|_| Zst).collect();
    let mut seen = 0;
    values.retain(|_| {
        seen += 1;
        seen % 2 == 0
    });
    assert_eq!(values.len(), 16);
    drop(values);
    ZST_DROPS.with(|x| assert_eq!(x.get(), 32));
}
