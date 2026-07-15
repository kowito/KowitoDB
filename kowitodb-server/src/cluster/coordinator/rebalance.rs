//! Cluster — membership-change rebalancing (split from coordinator.rs).
#![allow(clippy::result_large_err)]

use super::*;

impl Cluster {
    /// Rebalance object placement to match the current partitioning — call after
    /// a **membership change** (nodes added/removed) so each object lives on the
    /// node(s) that now own its id. Misplaced objects are copied to their correct
    /// owner replica set and removed from nodes that no longer own them. Returns
    /// the number of objects relocated. Best-effort and idempotent: re-running on
    /// a balanced cluster moves nothing.
    pub async fn rebalance(&self) -> Result<usize, Status> {
        let mut moved = 0usize;
        for (i, node) in self.nodes.iter().enumerate() {
            if !self.is_healthy(i) {
                continue;
            }
            let listed = match node
                .list(proto::ListRequest {
                    offset: 0,
                    limit: u32::MAX,
                })
                .await
            {
                Ok(r) => r,
                Err(_) => {
                    self.set_health(i, false);
                    continue;
                }
            };
            for obj in listed.objects {
                let Ok(id) = parse_id(&obj.id) else { continue };
                let owners = self.replicas_for_id(id);
                if owners.contains(&i) {
                    continue; // correctly placed on this node
                }
                // Relocate: write to the correct owners, then drop from this node
                // — but ONLY if at least one owner write succeeded, so a transient
                // owner failure can never delete the last copy (data loss).
                let insert = knowledge_to_insert_req(&obj);
                let mut acks = 0usize;
                for &o in &owners {
                    if self.nodes[o].insert(insert.clone()).await.is_ok() {
                        acks += 1;
                    }
                }
                if acks == 0 {
                    debug!(
                        "rebalance: no owner accepted {}; keeping source copy",
                        obj.id
                    );
                    continue;
                }
                if self.nodes[i]
                    .delete(proto::DeleteRequest { id: obj.id.clone() })
                    .await
                    .is_ok()
                {
                    moved += 1;
                    debug!("rebalance: moved {} off node {i}", obj.id);
                }
            }
        }
        info!("rebalance complete: {moved} object(s) relocated");
        Ok(moved)
    }
}
