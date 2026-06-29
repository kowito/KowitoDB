//! Benchmark embedvec on the SAME shared dataset as KowitoDB/Qdrant/Milvus.
//! Reads dataset.bin (written by `bench_compare`) and reports recall@k + QPS.
//!
//!   cargo run --release -- ../../../dataset.bin
//! Env: CMP_M, CMP_EFC, CMP_EFS (comma list).

use std::collections::HashSet;
use std::time::Instant;

use embedvec::{Distance, EmbedVec, Quantization};

struct Dataset {
    dim: usize,
    k: usize,
    vectors: Vec<Vec<f32>>,
    queries: Vec<Vec<f32>>,
    gt: Vec<Vec<u32>>,
}

fn rd_u32(b: &[u8], off: &mut usize) -> u32 {
    let v = u32::from_le_bytes(b[*off..*off + 4].try_into().unwrap());
    *off += 4;
    v
}

fn read_dataset(path: &str) -> Dataset {
    let b = std::fs::read(path).expect("read dataset.bin");
    assert_eq!(&b[0..4], b"KVDB", "bad dataset magic");
    let mut off = 4;
    let (n, dim, q, k) = (
        rd_u32(&b, &mut off) as usize,
        rd_u32(&b, &mut off) as usize,
        rd_u32(&b, &mut off) as usize,
        rd_u32(&b, &mut off) as usize,
    );
    let mut rd_vecs = |count: usize| -> Vec<Vec<f32>> {
        (0..count)
            .map(|_| {
                (0..dim)
                    .map(|_| {
                        let v = f32::from_le_bytes(b[off..off + 4].try_into().unwrap());
                        off += 4;
                        v
                    })
                    .collect()
            })
            .collect()
    };
    let vectors = rd_vecs(n);
    let queries = rd_vecs(q);
    let gt = (0..q)
        .map(|_| (0..k).map(|_| rd_u32(&b, &mut off)).collect())
        .collect();
    Dataset {
        dim,
        k,
        vectors,
        queries,
        gt,
    }
}

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

#[tokio::main]
async fn main() {
    let path = std::env::args().nth(1).unwrap_or_else(|| "dataset.bin".into());
    let ds = read_dataset(&path);
    let m = env_usize("CMP_M", 16);
    let efc = env_usize("CMP_EFC", 128);
    let ef_list: Vec<usize> = std::env::var("CMP_EFS")
        .unwrap_or_else(|_| "32,64,128,256".into())
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();

    // Build the index. Store the original vector index in metadata so results
    // map back to ground-truth indices regardless of embedvec's id scheme.
    // Use the builder with quantization OFF (exact f32) for a fair recall
    // comparison — `EmbedVec::new` defaults to lattice quantization.
    let mut db = EmbedVec::builder()
        .dimension(ds.dim)
        .metric(Distance::Cosine)
        .m(m)
        .ef_construction(efc)
        .quantization(Quantization::None)
        .build()
        .await
        .expect("create EmbedVec");
    let t = Instant::now();
    for (i, v) in ds.vectors.iter().enumerate() {
        db.add(v, serde_json::json!({ "i": i }))
            .await
            .expect("add");
    }
    eprintln!(
        "# embedvec build: {} vectors in {:.2}s",
        ds.vectors.len(),
        t.elapsed().as_secs_f64()
    );


    println!("system,ef_search,recall@{},qps_1thread,p50_us,p95_us", ds.k);
    for ef in ef_list {
        let mut hits = 0usize;
        let mut lat: Vec<u128> = Vec::with_capacity(ds.queries.len());
        for (qi, query) in ds.queries.iter().enumerate() {
            let t = Instant::now();
            let res = db.search(query, ds.k, ef, None).await.expect("search");
            lat.push(t.elapsed().as_micros());
            let truth: HashSet<u32> = ds.gt[qi].iter().copied().collect();
            for hit in &res {
                if let Some(i) = hit.payload.get("i").and_then(|v| v.as_u64()) {
                    if truth.contains(&(i as u32)) {
                        hits += 1;
                    }
                }
            }
        }
        lat.sort_unstable();
        let recall = hits as f64 / (ds.queries.len() * ds.k) as f64;
        let mean = lat.iter().sum::<u128>() as f64 / lat.len() as f64;
        let qps = 1_000_000.0 / mean;
        let p50 = lat[lat.len() / 2];
        let p95 = lat[(lat.len() as f64 * 0.95) as usize];
        println!("embedvec,{ef},{recall:.4},{qps:.0},{p50},{p95}");
    }
}
