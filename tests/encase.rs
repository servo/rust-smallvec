use smallvec::SmallVec;

#[test]
fn test_encase() {
    let v = SmallVec::<f32, 3>::from([1.43_f32, 4.05, 4.55]);

    let mut buf = [255u8; 12];
    let mut s_buf = encase::StorageBuffer::new(&mut buf);

    s_buf.write(&v).unwrap();
    assert_eq!(&s_buf.as_ref()[0..][..4], &v[0].to_le_bytes());
    assert_eq!(&s_buf.as_ref()[4..][..4], &v[1].to_le_bytes());
    assert_eq!(&s_buf.as_ref()[8..][..4], &v[2].to_le_bytes());

    let mut v_out: SmallVec<f32, 3> = SmallVec::new();
    s_buf.read(&mut v_out).unwrap();
    assert_eq!(v, v_out);

    assert_eq!(v, s_buf.create::<SmallVec<f32, 3>>().unwrap());
}
