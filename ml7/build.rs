// ASTRA7.md §5: on Linux, torch-sys's build.rs asks the linker to link
// libtorch_cuda.so, but the default --as-needed linker behavior silently
// drops that link because nothing in a typical crate references a CUDA
// symbol directly at link time. Result: the binary builds and runs fine,
// but tch/burn-tch report "no CUDA" / just run on CPU, with zero error.
//
// Only relevant when the `tch` feature (burn-tch) is active. Cargo sets
// CARGO_FEATURE_<NAME> env vars (uppercased, hyphens -> underscores) for
// build scripts, so CARGO_FEATURE_TCH is set iff the `tch` feature is on.
fn main() {
    if std::env::var("CARGO_FEATURE_TCH").is_err() {
        return;
    }

    let libtorch = std::env::var("LIBTORCH")
        .expect("LIBTORCH env var must be set when building with --features tch (see ASTRA7.md §5)");
    let lib = format!("{libtorch}/lib");

    println!("cargo:rustc-link-search=native={lib}");
    println!("cargo:rustc-link-arg=-Wl,--no-as-needed");
    println!("cargo:rustc-link-arg=-ltorch_cuda");
    println!("cargo:rustc-link-arg=-Wl,--as-needed");
}
