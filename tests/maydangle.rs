use smallvec::SmallVec;

#[derive(Default)]
struct MyData<'a> {
    arr: SmallVec<&'a u8, 5>,
    dat: u8
}

#[test]
fn may_dangle_allows_borrow_of_another_field() {
    let mut dat = MyData::default();
    dat.arr.push(&dat.dat);
}
