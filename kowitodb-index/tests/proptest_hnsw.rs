//! Property-based tests for the HNSW index. Random insert/remove/update
//! sequences are mirrored against a reference map; invariants that must hold
//! regardless of the (approximate) graph shape are asserted. This is the kind of
//! test that catches remove/update churn bugs (e.g. a stale entry point or an
//! edge pointing at a removed node) that example-based tests miss.

use std::collections::HashMap;

use kowitodb_index::{HnswIndex, HnswParams};
use proptest::prelude::*;
use uuid::Uuid;

const DIM: usize = 8;

#[derive(Debug, Clone)]
enum Op {
    Insert(u8, Vec<f32>),
    Remove(u8),
}

fn op_strategy() -> impl Strategy<Value = Op> {
    // Small id space so removes/updates frequently hit existing ids.
    prop_oneof![
        (0u8..16, prop::collection::vec(-1.0f32..1.0, DIM)).prop_map(|(id, v)| Op::Insert(id, v)),
        (0u8..16).prop_map(Op::Remove),
    ]
}

fn uid(id: u8) -> Uuid {
    Uuid::from_u128(id as u128 + 1)
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(96))]

    /// After any sequence of inserts/updates/removes the index stays consistent
    /// with a reference: same count, never returns a removed id, and survives a
    /// save/load round-trip with identical results.
    #[test]
    fn hnsw_consistent_under_churn(ops in prop::collection::vec(op_strategy(), 0..150)) {
        let idx = HnswIndex::new(HnswParams {
            m: 8,
            ef_construction: 64,
            ef_search: 64,
            ..Default::default()
        });
        let mut reference: HashMap<u8, Vec<f32>> = HashMap::new();

        for op in &ops {
            match op {
                Op::Insert(id, v) => {
                    idx.insert(uid(*id), v.clone());
                    reference.insert(*id, v.clone());
                }
                Op::Remove(id) => {
                    idx.remove(uid(*id));
                    reference.remove(id);
                }
            }
        }

        // 1. Count matches the reference exactly.
        prop_assert_eq!(idx.len(), reference.len());

        // 2. A search never returns a removed id (edge cleanup is correct), and
        //    every returned id is live. Probe with a few queries.
        let live: std::collections::HashSet<Uuid> = reference.keys().map(|&id| uid(id)).collect();
        let probes: Vec<Vec<f32>> = reference.values().take(3).cloned().collect();
        for q in probes.iter().chain(std::iter::once(&vec![0.0f32; DIM])) {
            for (rid, _) in idx.search(q, 16) {
                prop_assert!(live.contains(&rid), "search returned a non-live id {rid}");
            }
        }

        // 3. save/load is deterministic and preserves results.
        if let Some(q) = reference.values().next() {
            let path = std::env::temp_dir().join(format!("pt-hnsw-{}.bin", Uuid::new_v4()));
            idx.save(&path).unwrap();
            let loaded = HnswIndex::load(&path).unwrap().expect("snapshot loads");
            prop_assert_eq!(loaded.len(), idx.len());
            let a: Vec<Uuid> = idx.search(q, 5).into_iter().map(|(id, _)| id).collect();
            let b: Vec<Uuid> = loaded.search(q, 5).into_iter().map(|(id, _)| id).collect();
            prop_assert_eq!(a, b, "results must survive save/load");
            let _ = std::fs::remove_file(&path);
        }
    }

    /// With inserts only (no churn), every inserted vector finds itself — the
    /// graph stays navigable. (ef_search ≥ node count makes search exhaustive on
    /// these small sets, so this is exact, not probabilistic.)
    #[test]
    fn hnsw_inserts_are_all_findable(
        items in prop::collection::vec(prop::collection::vec(-1.0f32..1.0, DIM), 1..40)
    ) {
        let idx = HnswIndex::new(HnswParams {
            m: 8,
            ef_construction: 80,
            ef_search: 80,
            ..Default::default()
        });
        let ids: Vec<Uuid> = (0..items.len()).map(|i| Uuid::from_u128(i as u128 + 1)).collect();
        for (id, v) in ids.iter().zip(&items) {
            idx.insert(*id, v.clone());
        }
        prop_assert_eq!(idx.len(), items.len());
        for (id, v) in ids.iter().zip(&items) {
            let found = idx.search(v, items.len()).into_iter().any(|(rid, _)| rid == *id);
            prop_assert!(found, "inserted id {id} not findable among its own neighbors");
        }
    }
}
