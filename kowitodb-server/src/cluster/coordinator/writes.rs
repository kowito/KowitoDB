//! Cluster — write path (insert / batch / update / delete + replica quorum) (split from coordinator.rs).
#![allow(clippy::result_large_err)]

use super::*;

impl Cluster {
    pub async fn insert(
        &self,
        mut req: proto::InsertRequest,
    ) -> Result<proto::InsertResponse, Status> {
        let id = parse_or_new_id(req.id.as_deref());
        req.id = Some(id.to_string());
        self.write_to_replicas(&self.replicas_for_id(id), |n| {
            let r = req.clone();
            async move { n.insert(r).await.map(|_| ()) }
        })
        .await?;
        Ok(proto::InsertResponse { id: id.to_string() })
    }

    pub async fn remember(
        &self,
        mut req: proto::RememberRequest,
    ) -> Result<proto::RememberResponse, Status> {
        let id = parse_or_new_id(req.id.as_deref());
        req.id = Some(id.to_string());
        self.write_to_replicas(&self.replicas_for_id(id), |n| {
            let r = req.clone();
            async move { n.remember(r).await.map(|_| ()) }
        })
        .await?;
        Ok(proto::RememberResponse { id: id.to_string() })
    }

    pub async fn batch_insert(
        &self,
        req: proto::BatchInsertRequest,
    ) -> Result<proto::BatchInsertResponse, Status> {
        // Assign ids and record each id's replica set, then group items into
        // each node's sub-batch.
        let mut ids = Vec::with_capacity(req.items.len());
        let mut replica_sets: Vec<(String, Vec<usize>)> = Vec::with_capacity(req.items.len());
        let mut groups: HashMap<usize, Vec<proto::InsertRequest>> = HashMap::new();
        for mut item in req.items {
            let id = parse_or_new_id(item.id.as_deref());
            item.id = Some(id.to_string());
            let replicas = self.replicas_for_id(id);
            for &node in &replicas {
                groups.entry(node).or_default().push(item.clone());
            }
            ids.push(id.to_string());
            replica_sets.push((id.to_string(), replicas));
        }

        // Send each node's sub-batch and track which nodes acked.
        let mut failed: std::collections::HashSet<usize> = std::collections::HashSet::new();
        for (node, items) in groups {
            let req = proto::BatchInsertRequest { items };
            match self.nodes[node].batch_insert(req).await {
                Ok(_) => self.set_health(node, true),
                Err(e) => {
                    self.set_health(node, false);
                    warn!("batch_insert on node {node} failed: {e}");
                    failed.insert(node);
                }
            }
        }

        // Enforce the write quorum per object (same durability contract as the
        // single `insert`): every item must have reached `write_quorum` replicas.
        for (id, replicas) in &replica_sets {
            let quorum = self.write_quorum.clamp(1, replicas.len().max(1));
            let acks = replicas.iter().filter(|r| !failed.contains(r)).count();
            if acks < quorum {
                return Err(Status::unavailable(format!(
                    "batch_insert: write quorum not met for {id} ({acks}/{quorum} acks)"
                )));
            }
        }
        Ok(proto::BatchInsertResponse { ids })
    }

    /// Run `f` on each replica; succeed once `write_quorum` replicas ack.
    pub(crate) async fn write_to_replicas<F, Fut>(
        &self,
        replicas: &[usize],
        f: F,
    ) -> Result<(), Status>
    where
        F: Fn(Arc<dyn ClusterNode>) -> Fut,
        Fut: std::future::Future<Output = Result<(), Status>>,
    {
        let quorum = self.write_quorum.clamp(1, replicas.len().max(1));
        let mut acks = 0usize;
        let mut last_err = None;
        for &i in replicas {
            match f(self.nodes[i].clone()).await {
                Ok(()) => {
                    self.set_health(i, true);
                    acks += 1;
                }
                Err(e) => {
                    self.set_health(i, false);
                    warn!("write to replica {i} failed: {e}");
                    last_err = Some(e);
                }
            }
        }
        if acks >= quorum {
            Ok(())
        } else {
            Err(last_err.unwrap_or_else(|| {
                Status::unavailable(format!("write quorum not met: {acks}/{quorum} acks"))
            }))
        }
    }

    pub async fn update(&self, req: proto::UpdateRequest) -> Result<proto::UpdateResponse, Status> {
        let id = parse_id(&req.id)?;
        let mut out = proto::UpdateResponse {
            updated: false,
            version: 0,
        };
        for &i in &self.replicas_for_id(id) {
            if let Ok(resp) = self.nodes[i].update(req.clone()).await {
                if resp.updated {
                    out = resp;
                }
            }
        }
        Ok(out)
    }

    pub async fn delete(&self, req: proto::DeleteRequest) -> Result<proto::DeleteResponse, Status> {
        let id = parse_id(&req.id)?;
        let mut existed = false;
        for &i in &self.replicas_for_id(id) {
            if let Ok(resp) = self.nodes[i].delete(req.clone()).await {
                existed |= resp.existed;
            }
        }
        Ok(proto::DeleteResponse { existed })
    }
}
