use {
    smallvec::SmallVec,
    std::io::Write
};

#[test]
fn test_write() {
    let data = [1, 2, 3, 4, 5];

    let mut small_vec: SmallVec<u8, 2> = SmallVec::new();
    let len = small_vec.write(&data[..]).unwrap();
    assert_eq!(len, 5);
    assert_eq!(small_vec.as_ref(), data.as_ref());

    let mut small_vec: SmallVec<u8, 2> = SmallVec::new();
    small_vec.write_all(&data[..]).unwrap();
    assert_eq!(small_vec.as_ref(), data.as_ref());
}
