# alignment-rs

This is a Rust binding for alignment-cpp. Given the design of the C interface,
it should be relatively stable and not need changing often.

## Example

```rust
fn main() {
    let model = register_fingerveins(376, &left_image, &right_image);

    let comp = ImageModelComparator::new(0.55, 376, &left_image, &right_image);

    assert!(comp.compare_with_model(&model))
}
```
