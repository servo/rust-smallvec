rust-smallvec
=============

> **⚠️ Note:**
> This is the code for smallvec 2.0, which is not yet ready for release.  For
> details about the changes in version 2.0, please see [#183], [#240], and [#284].
>
> The source code for the latest smallvec 1.x.y release can be found on the
> [v1 branch].  Bug fixes for smallvec 1 should be based on that branch, while
> new feature development should go on the v2 branch.

[v1 branch]: https://github.com/servo/rust-smallvec/tree/v1
[#183]: https://github.com/servo/rust-smallvec/issues/183
[#240]: https://github.com/servo/rust-smallvec/issues/240
[#284]: https://github.com/servo/rust-smallvec/issues/284

## About smallvec

[Documentation](https://docs.rs/smallvec/)

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
