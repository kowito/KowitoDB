//! HNSW vector-index benchmark: build throughput, query latency, and recall.
//!
//! Generates a synthetic set of unit-norm random vectors, builds the HNSW
//! index, then measures search latency percentiles and recall@k against
//! brute-force cosine ground truth. Deterministic (fixed seed) and offline.
//!
//! Run (release strongly recommended):
//!   cargo run --release -p kowitodb-index --example bench_hnsw
//! Override defaults via env: BENCH_N, BENCH_DIM, BENCH_Q, BENCH_K, BENCH_EF.

use std::time::Instant;

use kowitodb_index::{HnswIndex, HnswParams, ShardedHnswIndex};
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

/// Brute-force top-k by cosine (dot product on unit vectors).
fn brute_force_topk(query: &[f32], data: &[(Uuid, Vec<f32>)], k: usize) -> Vec<Uuid> {
    let mut scored: Vec<(f32, Uuid)> = data
        .iter()
        .map(|(id, v)| {
            let dot: f32 = query.iter().zip(v).map(|(a, b)| a * b).sum();
            (dot, *id)
        })
        .collect();
    scored.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    scored.into_iter().take(k).map(|(_, id)| id).collect()
}

fn percentile(sorted_us: &[u128], p: f64) -> u128 {
    if sorted_us.is_empty() {
        return 0;
    }
    let idx = ((p / 100.0) * (sorted_us.len() as f64 - 1.0)).round() as usize;
    sorted_us[idx.min(sorted_us.len() - 1)]
}

fn main() {
    let n = env_usize("BENCH_N", 10_000);
    let dim = env_usize("BENCH_DIM", 384);
    let queries = env_usize("BENCH_Q", 200);
    let k = env_usize("BENCH_K", 10);
    // Default matches the shipped HnswParams::default() ef_search.
    let ef_search = env_usize("BENCH_EF", HnswParams::default().ef_search);

    println!("KowitoDB HNSW benchmark");
    println!("  vectors={n}  dim={dim}  queries={queries}  k={k}  ef_search={ef_search}");
    if cfg!(debug_assertions) {
        println!("  WARNING: debug build — run with --release for representative numbers.\n");
    } else {
        println!();
    }

    let mut rng = StdRng::seed_from_u64(42);

    // Generate dataset.
    let data: Vec<(Uuid, Vec<f32>)> = (0..n)
        .map(|i| {
            (
                Uuid::from_u128(i as u128 + 1),
                random_unit_vec(&mut rng, dim),
            )
        })
        .collect();

    // Build the index.
    let params = HnswParams {
        ef_search,
        ..Default::default()
    };
    let index = HnswIndex::new(params);
    let build_start = Instant::now();
    for (id, v) in &data {
        index.insert(*id, v.clone());
    }
    let build_elapsed = build_start.elapsed();
    let build_qps = n as f64 / build_elapsed.as_secs_f64();

    // Query set (fresh random vectors).
    let query_vecs: Vec<Vec<f32>> = (0..queries)
        .map(|_| random_unit_vec(&mut rng, dim))
        .collect();

    // Measure latency + recall.
    let mut latencies_us: Vec<u128> = Vec::with_capacity(queries);
    let mut hits = 0usize;
    let mut total = 0usize;
    for q in &query_vecs {
        let t = Instant::now();
        let approx = index.search(q, k);
        latencies_us.push(t.elapsed().as_micros());

        let truth = brute_force_topk(q, &data, k);
        let truth_set: std::collections::HashSet<Uuid> = truth.into_iter().collect();
        hits += approx
            .iter()
            .filter(|(id, _)| truth_set.contains(id))
            .count();
        total += k;
    }
    latencies_us.sort_unstable();

    let recall = hits as f64 / total as f64;
    let mean_us = latencies_us.iter().sum::<u128>() as f64 / latencies_us.len() as f64;
    let qps = 1_000_000.0 / mean_us;

    println!("Build:");
    println!(
        "  {n} vectors in {:.2}s  ({:.0} inserts/s)",
        build_elapsed.as_secs_f64(),
        build_qps
    );
    println!("Search (k={k}, ef_search={ef_search}):");
    println!("  recall@{k}: {:.1}%", recall * 100.0);
    println!(
        "  latency  mean={:.0}us  p50={}us  p95={}us  p99={}us",
        mean_us,
        percentile(&latencies_us, 50.0),
        percentile(&latencies_us, 95.0),
        percentile(&latencies_us, 99.0),
    );
    println!("  throughput ~{:.0} queries/s (single-threaded)", qps);

    // Compression variants (binary, Matryoshka) are evaluated on a *clustered*
    // dataset, which is representative of real embeddings (true neighbors are
    // meaningfully closer than random pairs). Pure random unit vectors are a
    // worst case for them — in high dimensions all pairwise distances
    // concentrate, leaving 1-bit codes and dimension-prefixes little to
    // separate, so they understate these techniques. Full/int8 distances are
    // exact, so those use the random set above.
    let n_clusters = 64usize;
    let centers: Vec<Vec<f32>> = (0..n_clusters)
        .map(|_| random_unit_vec(&mut rng, dim))
        .collect();
    let make_clustered = |rng: &mut StdRng, center: &[f32]| -> Vec<f32> {
        let mut v: Vec<f32> = center
            .iter()
            .map(|c| c + 0.25 * rng.gen_range(-1.0f32..1.0))
            .collect();
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for x in &mut v {
                *x /= norm;
            }
        }
        v
    };
    let cdata: Vec<(Uuid, Vec<f32>)> = (0..n)
        .map(|i| {
            let center = &centers[i % n_clusters];
            (
                Uuid::from_u128(i as u128 + 1),
                make_clustered(&mut rng, center),
            )
        })
        .collect();
    let cqueries: Vec<Vec<f32>> = (0..queries)
        .map(|i| make_clustered(&mut rng, &centers[i % n_clusters]))
        .collect();

    let eval = |index: &HnswIndex| -> (f64, f64) {
        let mut lat: Vec<u128> = Vec::with_capacity(queries);
        let mut hits = 0usize;
        for q in &cqueries {
            let t = Instant::now();
            let approx = index.search(q, k);
            lat.push(t.elapsed().as_micros());
            let truth: std::collections::HashSet<Uuid> =
                brute_force_topk(q, &cdata, k).into_iter().collect();
            hits += approx.iter().filter(|(id, _)| truth.contains(id)).count();
        }
        let mean = lat.iter().sum::<u128>() as f64 / lat.len() as f64;
        (hits as f64 / (queries * k) as f64, mean)
    };

    // Full-precision baseline on the clustered set (for a like-for-like recall
    // comparison with the variants below).
    let cfull = HnswIndex::new(HnswParams {
        ef_search,
        ..Default::default()
    });
    for (id, v) in &cdata {
        cfull.insert(*id, v.clone());
    }
    let (crec, cmean) = eval(&cfull);
    println!("\n— Compression variants (clustered data, n={n}) —");
    println!(
        "Full f32 baseline:  recall@{k}: {:.1}%   mean={:.0}us",
        crec * 100.0,
        cmean
    );

    let bin = HnswIndex::new(HnswParams {
        ef_search,
        binary_quantize: true,
        ..Default::default()
    });
    for (id, v) in &cdata {
        bin.insert(*id, v.clone());
    }
    let (brec, bmean) = eval(&bin);
    println!("Binary 1-bit RaBitQ (~32x less memory, popcount Hamming nav + refine):");
    println!(
        "  recall@{k}: {:.1}%   latency mean={:.0}us  ({:.2}x full)",
        brec * 100.0,
        bmean,
        bmean / cmean
    );

    let cdim = (dim / 4).max(1);
    let mat = HnswIndex::new(HnswParams {
        ef_search,
        coarse_dim: Some(cdim),
        ..Default::default()
    });
    for (id, v) in &cdata {
        mat.insert(*id, v.clone());
    }
    let (mrec, mmean) = eval(&mat);
    println!("Matryoshka adaptive (navigate {cdim}/{dim} dims, full-dim refine):");
    println!(
        "  recall@{k}: {:.1}%   latency mean={:.0}us  ({:.2}x full)",
        mrec * 100.0,
        mmean,
        mmean / cmean
    );

    // Concurrent throughput: queries run in parallel under the index's RwLock
    // read locks, so single-node QPS scales with cores.
    let threads = env_usize(
        "BENCH_THREADS",
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4),
    );
    if threads > 1 {
        let rounds = 5usize;
        let start = Instant::now();
        let total: usize = std::thread::scope(|scope| {
            let handles: Vec<_> = (0..threads)
                .map(|_| {
                    scope.spawn(|| {
                        let mut count = 0usize;
                        for _ in 0..rounds {
                            for q in &query_vecs {
                                let _ = index.search(q, k);
                                count += 1;
                            }
                        }
                        count
                    })
                })
                .collect();
            handles.into_iter().map(|h| h.join().unwrap()).sum()
        });
        let agg_qps = total as f64 / start.elapsed().as_secs_f64();
        println!("Concurrent ({threads} threads):");
        println!(
            "  aggregate throughput ~{:.0} queries/s ({:.1}x single-threaded)",
            agg_qps,
            agg_qps / qps
        );
    }

    // Parallel build via the sharded index (one thread per shard).
    let shards = env_usize("BENCH_SHARDS", threads.clamp(1, 16));
    if shards > 1 {
        let sharded = ShardedHnswIndex::new(
            shards,
            HnswParams {
                ef_search,
                ..Default::default()
            },
        );
        let items: Vec<_> = data.iter().map(|(id, v)| (*id, v.clone())).collect();
        let t = Instant::now();
        sharded.build_parallel(items);
        let pbuild = t.elapsed();

        let mut hits = 0usize;
        let mut total = 0usize;
        for q in &query_vecs {
            let approx = sharded.search(q, k);
            let truth: std::collections::HashSet<Uuid> =
                brute_force_topk(q, &data, k).into_iter().collect();
            hits += approx.iter().filter(|(id, _)| truth.contains(id)).count();
            total += k;
        }

        println!("Parallel build ({shards} shards):");
        println!(
            "  {n} vectors in {:.2}s  ({:.0} inserts/s, {:.1}x serial build)",
            pbuild.as_secs_f64(),
            n as f64 / pbuild.as_secs_f64(),
            build_elapsed.as_secs_f64() / pbuild.as_secs_f64(),
        );
        println!(
            "  sharded recall@{k}: {:.1}%",
            hits as f64 / total as f64 * 100.0
        );

        // int8-quantized index: ~4x less vector memory, measured recall.
        let qindex = ShardedHnswIndex::new(
            shards,
            HnswParams {
                ef_search,
                quantize: true,
                ..Default::default()
            },
        );
        qindex.build_parallel(data.iter().map(|(id, v)| (*id, v.clone())).collect());
        let mut qhits = 0usize;
        let mut qtotal = 0usize;
        for q in &query_vecs {
            let approx = qindex.search(q, k);
            let truth: std::collections::HashSet<Uuid> =
                brute_force_topk(q, &data, k).into_iter().collect();
            qhits += approx.iter().filter(|(id, _)| truth.contains(id)).count();
            qtotal += k;
        }
        println!("Quantized int8 ({shards} shards, ~4x less vector memory):");
        println!("  recall@{k}: {:.1}%", qhits as f64 / qtotal as f64 * 100.0);
    }
}
