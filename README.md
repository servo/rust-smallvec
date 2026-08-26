rust-smallvec
=============

> [!IMPORTANT]
> This branch contains the code for SmallVec v2, which is not yet ready for release.
> If your code is using any of the alpha version of SmallVec, beware that the API might
> change between versions.
>
> The status of v2 can be tracked in:
> - [its tracking issue](https://github.com/servo/rust-smallvec/issues/425)
> - [the wiki spec](https://github.com/servo/rust-smallvec/wiki)
>
> The source code for the latest smallvec 1.x.y release can be found on the
> [v1 branch](https://github.com/servo/rust-smallvec/tree/v1).  
> Bug fixes for smallvec v1 should be based on that branch, while
> new feature development should go on the v2 branch.

## About smallvec

[Documentation](https://docs.rs/smallvec/)

[Wiki](https://github.com/servo/rust-smallvec/wiki)

[Release notes](https://github.com/servo/rust-smallvec/releases)

"Small vector" optimization for Rust: store up to a small number of items on the stack

## Example

```rust
use smallvec::{SmallVec, smallvec};
    
// This SmallVec can hold up to 4 items on the stack:
let mut v: SmallVec<i32, 4> = smallvec![1, 2, 3, 4];

// It will automatically move its contents to the heap if
// contains more than four items:
v.push(5);

// SmallVec points to a slice, so you can use normal slice
// indexing and other methods to access its contents:
v[0] = v[1] + v[2];
v.sort();
```
