fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Use the vendored `protoc` so a fresh clone builds without anyone having to
    // `brew install protobuf` / `apt-get install protobuf-compiler` first. Falls
    // back to a system protoc (`PROTOC` env / `PATH`) if this platform has no
    // prebuilt binary.
    if std::env::var_os("PROTOC").is_none() {
        if let Ok(protoc) = protoc_bin_vendored::protoc_bin_path() {
            std::env::set_var("PROTOC", protoc);
        }
    }

    let out_dir = std::path::PathBuf::from(std::env::var("OUT_DIR")?);
    tonic_build::configure()
        .build_server(true)
        .build_client(true)
        // Emit the encoded file descriptor set so the server can expose gRPC
        // reflection (grpcurl, gRPC UIs, some load balancers).
        .file_descriptor_set_path(out_dir.join("kowitodb_descriptor.bin"))
        .compile_protos(&["../proto/kowitodb.proto"], &["../proto"])?;
    Ok(())
}
