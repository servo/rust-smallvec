#![allow(deprecated)]

use smallvec::{
    SmallVec,
    smallvec
};

fn check_eq<T: PartialEq + core::fmt::Debug + ?Sized>(a: &T, b: &T) {
    assert_eq!(a, b);
}

#[test]
fn smallvec() {
    let mut vec: SmallVec<i32, 2>;

    macro_rules! check {
        ($init:tt) => {
            vec = smallvec! $init;
            check_eq(&*vec, &*vec! $init);
        }
    }

    check!([0; 0]);
    check!([1; 1]);
    check!([2; 2]);
    check!([3; 3]);

    check!([]);
    check!([1]);
    check!([1, 2]);
    check!([1, 2, 3]);
}
