use {
    equivalent::Equivalent,
    smallvec::SmallVec
};

struct Key(&'static str);
struct Query(&'static str);

impl Equivalent<Key> for Query {
    fn equivalent(&self, key: &Key) -> bool {
        self.0.eq_ignore_ascii_case(key.0)
    }
}

#[test]
fn equivalent_slice() {
    let query: SmallVec<Query, 2> = SmallVec::from([Query("foo"), Query("bar")]);
    let key = [Key("FOO"), Key("BAR")];

    assert!(query.equivalent(&key[..]));
}

#[test]
fn equivalent_length_mismatch() {
    let query: SmallVec<Query, 2> = SmallVec::from([Query("foo")]);

    assert!(!query.equivalent(&[Key("foo"), Key("bar")][..]));
}
