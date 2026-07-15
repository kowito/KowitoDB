//! Cluster — the Cluster coordinator (scatter-gather, quorum, rebalance) (split from the former cluster.rs god-file).
#![allow(clippy::result_large_err)]

use super::node::RemoteNode;
use super::*;

/// The distributed coordinator over a set of data nodes.
pub struct Cluster {
    nodes: Vec<Arc<dyn ClusterNode>>,
    /// Proactively-tracked health per node (aligned with `nodes`). Reads skip
    /// nodes marked unhealthy; the heartbeat and per-request outcomes update it.
    health: Vec<AtomicBool>,
    replication_factor: usize,
    /// Minimum replica acks required for a write to succeed (durability).
    write_quorum: usize,
}

mod reads;
mod rebalance;
mod writes;

impl Cluster {
    /// Build a cluster from already-constructed nodes (write_quorum = 1).
    pub fn new(nodes: Vec<Arc<dyn ClusterNode>>, replication_factor: usize) -> Self {
        let n = nodes.len().max(1);
        let health = (0..nodes.len()).map(|_| AtomicBool::new(true)).collect();
        Self {
            nodes,
            health,
            replication_factor: replication_factor.clamp(1, n),
            write_quorum: 1,
        }
    }

    pub(crate) fn is_healthy(&self, i: usize) -> bool {
        self.health[i].load(Ordering::Relaxed)
    }

    /// Update a node's health, logging up/down transitions.
    pub(crate) fn set_health(&self, i: usize, healthy: bool) {
        let prev = self.health[i].swap(healthy, Ordering::Relaxed);
        if prev != healthy {
            if healthy {
                info!("Cluster: node {i} recovered (healthy)");
            } else {
                warn!("Cluster: node {i} marked unhealthy");
            }
        }
    }

    /// Number of nodes currently considered healthy.
    pub fn healthy_count(&self) -> usize {
        (0..self.nodes.len())
            .filter(|&i| self.is_healthy(i))
            .count()
    }

    /// Probe every node once (cheap `stats` call) and update health. Run
    /// periodically by the gateway so down nodes are detected and recovered
    /// without waiting for a request to hit them.
    pub async fn heartbeat_once(&self) {
        for i in 0..self.nodes.len() {
            let ok = self.nodes[i].stats(proto::StatsRequest {}).await.is_ok();
            self.set_health(i, ok);
        }
    }

    /// Require `w` replica acks per write (clamped to the replication factor).
    /// `w >= ceil((R+1)/2)` gives majority-quorum durability.
    pub fn with_write_quorum(mut self, w: usize) -> Self {
        self.write_quorum = w.clamp(1, self.replication_factor);
        self
    }

    /// Connect to peer data nodes over gRPC.
    pub async fn connect(
        peers: &[String],
        replication_factor: usize,
        write_quorum: usize,
        api_key: Option<String>,
    ) -> anyhow::Result<Self> {
        if peers.is_empty() {
            anyhow::bail!("a cluster needs at least one peer node");
        }
        let mut nodes: Vec<Arc<dyn ClusterNode>> = Vec::with_capacity(peers.len());
        for peer in peers {
            let node = RemoteNode::connect(peer.clone(), api_key.as_deref()).await?;
            info!("Cluster: connected to data node {}", node.addr());
            nodes.push(Arc::new(node));
        }
        Ok(Self::new(nodes, replication_factor).with_write_quorum(write_quorum))
    }

    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    pub(crate) fn replicas(&self, owner: usize) -> Vec<usize> {
        let n = self.nodes.len();
        (0..self.replication_factor)
            .map(|i| (owner + i) % n)
            .collect()
    }

    pub(crate) fn replicas_for_id(&self, id: ObjectId) -> Vec<usize> {
        self.replicas((id.as_u128() % self.nodes.len() as u128) as usize)
    }

    pub(crate) fn replicas_for_key(&self, key: &str) -> Vec<usize> {
        let h = key.bytes().fold(1469598103934665603u64, |a, b| {
            (a ^ b as u64).wrapping_mul(1099511628211)
        });
        self.replicas((h % self.nodes.len() as u64) as usize)
    }
}
