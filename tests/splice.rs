use smallvec::{smallvec, SmallVec};

#[test]
fn test_splice() {
    let mut v: SmallVec<[_; 1]> = smallvec![1, 2, 3, 4, 5];
    let a = [10, 11, 12];
    v.splice(2..4, a.iter().cloned());
    assert_eq!(&v[..], &[1, 2, 10, 11, 12, 5]);
    v.splice(1..3, Some(20));
    assert_eq!(&v[..], &[1, 20, 11, 12, 5]);
}

#[test]
fn test_splice_inclusive_range() {
    let mut v: SmallVec<[_; 1]> = smallvec![1, 2, 3, 4, 5];
    let a = [10, 11, 12];
    let t1: SmallVec<[_; 2]> = v.splice(2..=3, a.iter().cloned()).collect();
    assert_eq!(&v[..], &[1, 2, 10, 11, 12, 5]);
    assert_eq!(&t1[..], &[3, 4]);
    let t2: SmallVec<[_; 2]> = v.splice(1..=2, Some(20)).collect();
    assert_eq!(&v[..], &[1, 20, 11, 12, 5]);
    assert_eq!(&t2[..], &[2, 10]);
}

#[test]
#[should_panic]
fn test_splice_out_of_bounds() {
    let mut v: SmallVec<[_; 1]> = smallvec![1, 2, 3, 4, 5];
    let a = [10, 11, 12];
    v.splice(5..6, a.iter().cloned());
}

#[test]
#[should_panic]
fn test_splice_inclusive_out_of_bounds() {
    let mut v: SmallVec<[_; 1]> = smallvec![1, 2, 3, 4, 5];
    let a = [10, 11, 12];
    v.splice(5..=5, a.iter().cloned());
}

#[test]
fn test_splice_items_zero_sized() {
    let mut smallvec: SmallVec<[_; 1]> = smallvec![(), (), ()];
    let smallvec2 = vec![];
    let t: SmallVec<[_; 2]> = smallvec.splice(1..2, smallvec2.iter().cloned()).collect();
    assert_eq!(&smallvec[..], &[(), ()]);
    assert_eq!(&t[..], &[()]);
}

#[test]
fn test_splice_unbounded() {
    let mut smallvec: SmallVec<[_; 1]> = smallvec![1, 2, 3, 4, 5];
    let t: SmallVec<[_; 2]> = smallvec.splice(.., None).collect();
    assert_eq!(&smallvec[..], &[]);
    assert_eq!(&t[..], &[1, 2, 3, 4, 5]);
}

#[test]
fn test_splice_forget() {
    let mut v: SmallVec<[_; 1]> = smallvec![1, 2, 3, 4, 5];
    let a = [10, 11, 12];
    std::mem::forget(v.splice(2..4, a.iter().cloned()));
    assert_eq!(&v[..], &[1, 2]);
}
