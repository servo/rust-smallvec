use std::{
    env,
    env::var_os,
    path::PathBuf,
    process::Command
};

fn main() {
    println!("cargo:rerun-if-changed=syntax.rs");
    let cwd = PathBuf::from(var_os("CARGO_MANIFEST_DIR").unwrap());
    let out = PathBuf::from(var_os("OUT_DIR").unwrap());
    let status = Command::new(var_os("RUSTC").unwrap())
        .current_dir(&cwd)
        .args([
            "--crate-name=syntax",
            "--crate-type=proc-macro",
            "--edition=2024",
            "--extern=proc_macro",
            "./syntax.rs",
            "--out-dir"
        ])
        .arg(&out)
        .status()
        .unwrap();
    assert!(status.success());
    println!("cargo:rustc-link-search=all={}", out.display());
}
