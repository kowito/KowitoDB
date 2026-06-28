//! Apples-to-apples ANN benchmark for KowitoDB vs Qdrant vs Milvus.
//!
//! Generates a deterministic dataset (random unit vectors + queries +
//! brute-force ground truth), benchmarks the KowitoDB HNSW index on it, and
//! writes the dataset to a shared binary file so the Python harnesses
//! (`benchmarks/comparison/bench_qdrant.py`, `bench_milvus.py`) measure the
//! *same* vectors, queries, and ground truth at the *same* HNSW parameters.
//!
//! Run:
//!   cargo run --release -p kowitodb-index --example bench_compare -- dataset.bin
//! Env: CMP_N, CMP_DIM, CMP_Q, CMP_K, CMP_M, CMP_EFC, CMP_EFS (comma list).
//!
//! Fairness notes are in benchmarks/comparison/README.md. In short: matched
//! M / ef_construction / ef_search and identical data; KowitoDB is measured
//! in-process (library), Qdrant/Milvus as localhost services — so KowitoDB has
//! no network round-trip. The recall column is the apples-to-apples one.

use std::io::Write;
use std::time::Instant;

use kowitodb_index::{HnswIndex, HnswParams};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use uuid::Uuid;

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn random_unit_vec(rng: &mut StdRng, dim: usize) -> Vec<f32> {
    let mut v: Vec<f32> = (0..dim).map(|_| rng.gen_range(-1.0f32..1.0)).collect();
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in &mut v {
            *x /= norm;
        }
    }
    v
}

fn main() {
    let out_path = std::env::args().nth(1).unwrap_or_else(|| "dataset.bin".into());
    let n = env_usize("CMP_N", 50_000);
    let dim = env_usize("CMP_DIM", 128);
    let q = env_usize("CMP_Q", 1_000);
    let k = env_usize("CMP_K", 10);
    let m = env_usize("CMP_M", 16);
    let efc = env_usize("CMP_EFC", 128);
    let diversify = env_usize("CMP_DIVERSIFY", 0) != 0;
    let ef_list: Vec<usize> = std::env::var("CMP_EFS")
        .unwrap_or_else(|_| "32,64,128,256".into())
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();

    println!("# KowitoDB vs Qdrant vs Milvus — ANN comparison");
    println!("# dataset: n={n} dim={dim} queries={q} k={k}  HNSW: M={m} ef_construction={efc}");
    if cfg!(debug_assertions) {
        println!("# WARNING: debug build — use --release for representative numbers.");
    }

    // CMP_CLUSTERS>0 generates clustered data (representative of real
    // embeddings); 0 (default) is uniform random (a harder, structure-free case).
    let clusters = env_usize("CMP_CLUSTERS", 0);
    let mut rng = StdRng::seed_from_u64(1234);
    let (vectors, queries) = if clusters > 0 {
        let centers: Vec<Vec<f32>> = (0..clusters).map(|_| random_unit_vec(&mut rng, dim)).collect();
        let perturb = |rng: &mut StdRng, c: &[f32]| -> Vec<f32> {
            let mut v: Vec<f32> = c.iter().map(|x| x + 0.20 * rng.gen_range(-1.0f32..1.0)).collect();
            let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 0.0 {
                for x in &mut v {
                    *x /= norm;
                }
            }
            v
        };
        let vectors: Vec<Vec<f32>> =
            (0..n).map(|i| perturb(&mut rng, &centers[i % clusters])).collect();
        let queries: Vec<Vec<f32>> =
            (0..q).map(|i| perturb(&mut rng, &centers[i % clusters])).collect();
        (vectors, queries)
    } else {
        let vectors: Vec<Vec<f32>> = (0..n).map(|_| random_unit_vec(&mut rng, dim)).collect();
        let queries: Vec<Vec<f32>> = (0..q).map(|_| random_unit_vec(&mut rng, dim)).collect();
        (vectors, queries)
    };

    // Brute-force ground truth (top-k by cosine = dot on unit vectors).
    let ground_truth: Vec<Vec<u32>> = queries
        .iter()
        .map(|query| {
            let mut scored: Vec<(f32, u32)> = vectors
                .iter()
                .enumerate()
                .map(|(i, v)| {
                    let dot: f32 = query.iter().zip(v).map(|(a, b)| a * b).sum();
                    (dot, i as u32)
                })
                .collect();
            scored.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
            scored.into_iter().take(k).map(|(_, i)| i).collect()
        })
        .collect();

    // Write the shared dataset file for the Python harnesses.
    write_dataset(&out_path, &vectors, &queries, &ground_truth, k);
    println!("# wrote shared dataset to {out_path}\n");

    // ---- Benchmark KowitoDB ----
    // Point ids are the vector indices (encoded as UUIDs) so recall compares by
    // index, identically to Qdrant/Milvus which use the integer index as id.
    let index = HnswIndex::new(HnswParams {
        m,
        ef_construction: efc,
        ef_search: *ef_list.iter().max().unwrap_or(&64),
        diversify_neighbors: diversify,
        ..Default::default()
    });
    let t = Instant::now();
    for (i, v) in vectors.iter().enumerate() {
        index.insert(Uuid::from_u128(i as u128), v.clone());
    }
    let build_s = t.elapsed().as_secs_f64();
    println!(
        "KowitoDB build: {n} vectors in {build_s:.2}s ({:.0}/s)",
        n as f64 / build_s
    );

    println!("\nsystem,ef_search,recall@{k},qps_1thread,p50_us,p95_us");
    for &ef in &ef_list {
        // ef_search is baked into the index params, so build one index per ef
        // value to measure each operating point on the recall/QPS curve.
        let idx = HnswIndex::new(HnswParams {
            m,
            ef_construction: efc,
            ef_search: ef,
            diversify_neighbors: diversify,
            ..Default::default()
        });
        for (i, v) in vectors.iter().enumerate() {
            idx.insert(Uuid::from_u128(i as u128), v.clone());
        }

        let mut hits = 0usize;
        let mut lat: Vec<u128> = Vec::with_capacity(q);
        for (qi, query) in queries.iter().enumerate() {
            let t = Instant::now();
            let res = idx.search(query, k);
            lat.push(t.elapsed().as_micros());
            let truth: std::collections::HashSet<u32> =
                ground_truth[qi].iter().copied().collect();
            hits += res
                .iter()
                .filter(|(id, _)| truth.contains(&(id.as_u128() as u32)))
                .count();
        }
        lat.sort_unstable();
        let recall = hits as f64 / (q * k) as f64;
        let mean_us = lat.iter().sum::<u128>() as f64 / lat.len() as f64;
        let qps = 1_000_000.0 / mean_us;
        let p50 = lat[lat.len() / 2];
        let p95 = lat[(lat.len() as f64 * 0.95) as usize];
        println!("kowitodb,{ef},{recall:.4},{qps:.0},{p50},{p95}");
    }
}

fn write_dataset(path: &str, vectors: &[Vec<f32>], queries: &[Vec<f32>], gt: &[Vec<u32>], k: usize) {
    let n = vectors.len();
    let dim = vectors[0].len();
    let q = queries.len();
    let mut buf: Vec<u8> = Vec::new();
    buf.extend_from_slice(b"KVDB");
    for v in [n as u32, dim as u32, q as u32, k as u32] {
        buf.extend_from_slice(&v.to_le_bytes());
    }
    for vec in vectors {
        for &x in vec {
            buf.extend_from_slice(&x.to_le_bytes());
        }
    }
    for vec in queries {
        for &x in vec {
            buf.extend_from_slice(&x.to_le_bytes());
        }
    }
    for row in gt {
        for &idx in row {
            buf.extend_from_slice(&idx.to_le_bytes());
        }
    }
    let mut f = std::fs::File::create(path).expect("create dataset file");
    f.write_all(&buf).expect("write dataset");
}
