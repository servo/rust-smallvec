# rust-smallvec

> [!WARNING]
> smallvec v2 is on pre-release
>
> this means that there might be unexpected changes between versions. see the changelog
> for detailed breakdowns
>
> changes include, but are not limited to:
> - breaking API changes
> - deprecation of items
> - adding, removing or modifying features
> 
> beware that this is a pre-release and we can't ensure that there are no vulnerabilities
> or corner cases. if this feels like a substantial risk to you, please downgrade to the
> latest v1 version
> 
> we are looking for people to test this v2 version, so feel free to play around with
> smallvec and use it for your purposes if this warning is not a concern, and please file
> an issue on the repo if you find some unexpected behavior or have a request for a feature

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
