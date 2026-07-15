//! `HnswIndex` tests (split from the former hnsw.rs god-file).

use super::*;
use std::collections::HashSet;

#[test]
fn test_quantized_index_search() {
    // int8-quantized index: exact matches stay top-1 despite quantization.
    let idx = HnswIndex::new(HnswParams {
        m: 8,
        ef_construction: 50,
        ef_search: 50,
        quantize: true,
        ..Default::default()
    });
    let mut items = Vec::new();
    for i in 0..100 {
        let id = uuid::Uuid::new_v4();
        // Components in [-1, 1], matching the quantization assumption.
        let v: Vec<f32> = (0..16).map(|j| ((i * 7 + j * 3) as f32).sin()).collect();
        idx.insert(id, v.clone());
        items.push((id, v));
    }

    let (qid, qv) = &items[42];
    let results = idx.search(qv, 5);
    assert_eq!(results.len(), 5);
    assert_eq!(
        results[0].0, *qid,
        "exact match should remain top-1 under int8 quantization"
    );
}

#[test]
fn test_binary_quantized_search_recall() {
    // RaBitQ-style 1-bit index: exact matches stay near the top and
    // recall@10 vs brute force is reasonable despite ~32× compression.
    let idx = HnswIndex::new(HnswParams {
        m: 16,
        ef_construction: 100,
        ef_search: 100,
        binary_quantize: true,
        ..Default::default()
    });
    let mut items = Vec::new();
    for i in 0..200u32 {
        let id = uuid::Uuid::from_u128(i as u128 + 1);
        let v: Vec<f32> = (0..64)
            .map(|j| (((i * 13 + j * 7) as f32) * 0.1).sin())
            .collect();
        idx.insert(id, v.clone());
        items.push((id, v));
    }

    let (qid, qv) = &items[42];
    let res = idx.search(qv, 5);
    assert!(
        res.iter().take(3).any(|(id, _)| id == qid),
        "exact match should be near top-1 under binary quantization"
    );

    // Recall@10 vs brute-force ground truth over a few probes.
    let probes = [7usize, 99, 150];
    let (mut hit, mut total) = (0usize, 0usize);
    for &p in &probes {
        let q = &items[p].1;
        let mut bf: Vec<_> = items
            .iter()
            .map(|(id, v)| (*id, squared_dist(q, v)))
            .collect();
        bf.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
        let truth: HashSet<_> = bf.iter().take(10).map(|(id, _)| *id).collect();
        for (id, _) in idx.search(q, 10) {
            if truth.contains(&id) {
                hit += 1;
            }
            total += 1;
        }
    }
    let recall = hit as f32 / total as f32;
    assert!(recall >= 0.5, "binary recall@10 too low: {recall}");
}

#[test]
fn test_matryoshka_coarse_search_refines() {
    // Coarse (prefix-dim) navigation + full-dim refine: exact matches stay
    // top-1 and recall@10 vs brute force stays high because the final
    // ranking is computed at full dimension.
    let idx = HnswIndex::new(HnswParams {
        m: 16,
        ef_construction: 100,
        ef_search: 100,
        coarse_dim: Some(16), // navigate on the first 16 of 64 dims
        ..Default::default()
    });
    let mut items = Vec::new();
    for i in 0..200u32 {
        let id = uuid::Uuid::from_u128(i as u128 + 1);
        let v: Vec<f32> = (0..64)
            .map(|j| (((i * 13 + j * 7) as f32) * 0.1).sin())
            .collect();
        idx.insert(id, v.clone());
        items.push((id, v));
    }

    let (qid, qv) = &items[42];
    assert_eq!(
        idx.search(qv, 5)[0].0,
        *qid,
        "exact match should be top-1 after full-dim refine"
    );

    let probes = [7usize, 99, 150];
    let (mut hit, mut total) = (0usize, 0usize);
    for &p in &probes {
        let q = &items[p].1;
        let mut bf: Vec<_> = items
            .iter()
            .map(|(id, v)| (*id, squared_dist(q, v)))
            .collect();
        bf.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
        let truth: HashSet<_> = bf.iter().take(10).map(|(id, _)| *id).collect();
        for (id, _) in idx.search(q, 10) {
            if truth.contains(&id) {
                hit += 1;
            }
            total += 1;
        }
    }
    let recall = hit as f32 / total as f32;
    assert!(recall >= 0.7, "matryoshka recall@10 too low: {recall}");
}

#[test]
fn test_binary_rerank_improves_recall() {
    // Oversample→rescore with retained int8 vectors should recover recall
    // the 1-bit codes lose: rerank recall ≥ plain-binary recall, and high.
    let build = |rerank: bool| {
        let idx = HnswIndex::new(HnswParams {
            m: 16,
            ef_construction: 100,
            ef_search: 100,
            binary_quantize: true,
            binary_rerank: rerank,
            ..Default::default()
        });
        let mut items = Vec::new();
        for i in 0..200u32 {
            let id = uuid::Uuid::from_u128(i as u128 + 1);
            let v: Vec<f32> = (0..64)
                .map(|j| (((i * 13 + j * 7) as f32) * 0.1).sin())
                .collect();
            idx.insert(id, v.clone());
            items.push((id, v));
        }
        (idx, items)
    };

    let recall_of = |idx: &HnswIndex, items: &[(uuid::Uuid, Vec<f32>)]| -> f32 {
        let probes = [7usize, 42, 99, 150, 175];
        let (mut hit, mut total) = (0usize, 0usize);
        for &p in &probes {
            let q = &items[p].1;
            let mut bf: Vec<_> = items
                .iter()
                .map(|(id, v)| (*id, squared_dist(q, v)))
                .collect();
            bf.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
            let truth: HashSet<_> = bf.iter().take(10).map(|(id, _)| *id).collect();
            for (id, _) in idx.search(q, 10) {
                if truth.contains(&id) {
                    hit += 1;
                }
                total += 1;
            }
        }
        hit as f32 / total as f32
    };

    let (plain, items) = build(false);
    let (reranked, items2) = build(true);
    let plain_recall = recall_of(&plain, &items);
    let rerank_recall = recall_of(&reranked, &items2);

    assert!(
        rerank_recall >= plain_recall,
        "rerank ({rerank_recall}) should not hurt recall vs plain binary ({plain_recall})"
    );
    assert!(
        rerank_recall >= 0.85,
        "binary+int8-rerank recall too low: {rerank_recall}"
    );
}

#[test]
fn test_binary_quantized_save_load() {
    let idx = HnswIndex::new(HnswParams {
        m: 8,
        ef_construction: 50,
        ef_search: 50,
        binary_quantize: true,
        ..Default::default()
    });
    let mut items = Vec::new();
    for i in 0..60u32 {
        let id = uuid::Uuid::from_u128(i as u128 + 1);
        let v: Vec<f32> = (0..32)
            .map(|j| (((i * 11 + j * 5) as f32) * 0.1).sin())
            .collect();
        idx.insert(id, v.clone());
        items.push((id, v));
    }

    let path = std::env::temp_dir().join(format!("kowitodb-bq-{}.bin", uuid::Uuid::new_v4()));
    idx.save(&path).unwrap();
    let loaded = HnswIndex::load(&path).unwrap().expect("snapshot loads");

    // The rotation must survive the round-trip, so results match exactly.
    let q = &items[20].1;
    let before: Vec<_> = idx.search(q, 5).into_iter().map(|(id, _)| id).collect();
    let after: Vec<_> = loaded.search(q, 5).into_iter().map(|(id, _)| id).collect();
    assert_eq!(before, after, "binary search must survive save/load");
    let _ = std::fs::remove_file(&path);
}

#[test]
fn test_diversify_neighbors_builds_valid_graph() {
    // With the diversity heuristic on, the graph must still return exact
    // matches and rank by descending similarity.
    let idx = HnswIndex::new(HnswParams {
        m: 8,
        ef_construction: 64,
        ef_search: 64,
        diversify_neighbors: true,
        ..Default::default()
    });
    let mut items = Vec::new();
    for i in 0..150u32 {
        let id = uuid::Uuid::from_u128(i as u128 + 1);
        let v: Vec<f32> = (0..32)
            .map(|j| (((i * 13 + j * 7) as f32) * 0.1).sin())
            .collect();
        idx.insert(id, v.clone());
        items.push((id, v));
    }
    let (qid, qv) = &items[42];
    let res = idx.search(qv, 5);
    assert_eq!(res[0].0, *qid, "exact match should be top-1");
    for w in res.windows(2) {
        assert!(w[0].1 >= w[1].1, "results sorted by descending similarity");
    }
}

#[test]
fn test_hnsw_insert_and_search() {
    let idx = HnswIndex::new(HnswParams {
        m: 8,
        ef_construction: 50,
        ef_search: 20,
        ..Default::default()
    });

    // Insert 50 random vectors
    let mut ids = Vec::new();
    for i in 0..50 {
        let id = uuid::Uuid::new_v4();
        let vec: Vec<f32> = (0..16).map(|j| ((i * 7 + j * 3) as f32).sin()).collect();
        idx.insert(id, vec);
        ids.push(id);
    }

    // Search should return results
    let query: Vec<f32> = (0..16)
        .map(|j| (25.0 * 7.0 + j as f32 * 3.0).sin())
        .collect();
    let results = idx.search(&query, 5);
    assert_eq!(results.len(), 5);
    // Scores should be in descending order (similarity)
    for w in results.windows(2) {
        assert!(
            w[0].1 >= w[1].1,
            "Results should be sorted by descending score"
        );
    }
}

#[test]
fn test_hnsw_save_load_roundtrip() {
    let idx = HnswIndex::new(HnswParams {
        m: 8,
        ef_construction: 50,
        ef_search: 20,
        ..Default::default()
    });
    let mut ids = Vec::new();
    for i in 0..50 {
        let id = uuid::Uuid::new_v4();
        let vec: Vec<f32> = (0..16).map(|j| ((i * 7 + j * 3) as f32).sin()).collect();
        idx.insert(id, vec);
        ids.push(id);
    }

    let path = std::env::temp_dir().join(format!("kowitodb-hnsw-{}.bin", uuid::Uuid::new_v4()));
    idx.save(&path).unwrap();

    // Loading a missing file yields None.
    let missing = std::env::temp_dir().join("kowitodb-hnsw-does-not-exist.bin");
    assert!(HnswIndex::load(&missing).unwrap().is_none());

    let loaded = HnswIndex::load(&path)
        .unwrap()
        .expect("snapshot should load");
    assert_eq!(loaded.len(), idx.len());

    // The loaded index returns the same neighbors as the original.
    let query: Vec<f32> = (0..16)
        .map(|j| (25.0 * 7.0 + j as f32 * 3.0).sin())
        .collect();
    let before = idx.search(&query, 5);
    let after = loaded.search(&query, 5);
    let before_ids: Vec<_> = before.iter().map(|(id, _)| *id).collect();
    let after_ids: Vec<_> = after.iter().map(|(id, _)| *id).collect();
    assert_eq!(
        before_ids, after_ids,
        "search results must survive save/load"
    );

    let _ = std::fs::remove_file(&path);
}

#[test]
fn test_reinsert_preserves_search() {
    // Re-inserting existing ids triggers `remove_locked` (including on the
    // current entry point). Regression: removing the entry point used to
    // leave it at a low layer with `max_layer` stale, collapsing recall.
    let idx = HnswIndex::new(HnswParams {
        m: 8,
        ef_construction: 64,
        ef_search: 64,
        ..Default::default()
    });
    let mut items = Vec::new();
    for i in 0..120u32 {
        let id = uuid::Uuid::from_u128(i as u128 + 1);
        let v: Vec<f32> = (0..24)
            .map(|j| (((i * 13 + j * 7) as f32) * 0.1).sin())
            .collect();
        idx.insert(id, v.clone());
        items.push((id, v));
    }
    // Re-insert every item (each is an update → remove + re-add churn).
    for (id, v) in &items {
        idx.insert(*id, v.clone());
    }
    assert_eq!(idx.len(), 120, "updates must not change node count");
    let (qid, qv) = &items[42];
    assert_eq!(
        idx.search(qv, 1)[0].0,
        *qid,
        "exact match must stay top-1 after entry-point churn"
    );
}

#[test]
fn test_dimension_mismatch_is_rejected() {
    let idx = HnswIndex::new(HnswParams::default());
    idx.insert(uuid::Uuid::from_u128(1), vec![1.0, 0.0, 0.0]);
    assert_eq!(idx.dimension(), Some(3));
    // A wrong-dimension insert is skipped (not added).
    idx.insert(uuid::Uuid::from_u128(2), vec![1.0, 0.0]);
    idx.insert(uuid::Uuid::from_u128(3), vec![0.0, 1.0, 0.0, 0.0]);
    assert_eq!(idx.len(), 1, "mismatched-dim inserts must be skipped");
    // A wrong-dimension query returns nothing rather than garbage.
    assert!(idx.search(&vec![1.0, 0.0], 5).is_empty());
    assert_eq!(idx.search(&vec![1.0, 0.0, 0.0], 5).len(), 1);
}

#[test]
fn test_hnsw_empty_search() {
    let idx = HnswIndex::new(HnswParams::default());
    let results = idx.search(&vec![1.0, 2.0, 3.0], 5);
    assert!(results.is_empty());
}

#[test]
fn test_hnsw_remove() {
    let idx = HnswIndex::new(HnswParams {
        m: 4,
        ef_construction: 20,
        ef_search: 10,
        ..Default::default()
    });

    let id = uuid::Uuid::new_v4();
    idx.insert(id, vec![1.0, 0.0, 0.0]);
    idx.insert(uuid::Uuid::new_v4(), vec![0.0, 1.0, 0.0]);

    assert_eq!(idx.len(), 2);
    idx.remove(id);
    assert_eq!(idx.len(), 1);

    let results = idx.search(&vec![0.9, 0.1, 0.0], 3);
    assert_eq!(results.len(), 1);
}

#[test]
fn test_squared_dist() {
    let a = vec![0.0, 3.0, 4.0];
    let b = vec![0.0, 0.0, 0.0];
    // 3^2 + 4^2 = 25 (squared distance — no sqrt).
    assert!((squared_dist(&a, &b) - 25.0).abs() < 1e-6);
}
