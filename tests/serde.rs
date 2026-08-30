use {
    serde_test::{
        Token,
        assert_tokens
    },
    smallvec::SmallVec
};

#[test]
fn test_serde() {
    let mut small_vec: SmallVec<i32, 2> = SmallVec::new();
    assert_tokens(
        &small_vec,
        &[
            Token::Seq {
                len: Some(0)
            },
            Token::SeqEnd
        ]
    );
    small_vec.push(1);
    assert_tokens(
        &small_vec,
        &[
            Token::Seq {
                len: Some(1)
            },
            Token::I32(1),
            Token::SeqEnd
        ]
    );
    small_vec.extend([2, 3, 4]);
    assert_tokens(
        &small_vec,
        &[
            Token::Seq {
                len: Some(4)
            },
            Token::I32(1),
            Token::I32(2),
            Token::I32(3),
            Token::I32(4),
            Token::SeqEnd
        ]
    );
}
