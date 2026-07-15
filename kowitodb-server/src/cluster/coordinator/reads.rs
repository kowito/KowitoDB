//! Cluster — read path (scatter-gather search/ask/sql/list/stats + read-repair get) (split from coordinator.rs).
#![allow(clippy::result_large_err)]

use super::*;

impl Cluster {
    /// Read an object, with **read-repair + last-write-wins reconciliation**:
    /// query all healthy replicas, return the freshest copy (latest
    /// `updated_at`), and write that copy back to any replica that is missing it
    /// *or holds a staler/divergent copy* — so the cluster converges on the most
    /// recent version after a partial write or a divergence.
    pub async fn get(&self, req: proto::GetRequest) -> Result<proto::GetResponse, Status> {
        let id = parse_id(&req.id)?;
        let replicas: Vec<usize> = self
            .replicas_for_id(id)
            .into_iter()
            .filter(|&i| self.is_healthy(i))
            .collect();

        let futures = replicas.iter().map(|&i| {
            let node = self.nodes[i].clone();
            let req = req.clone();
            async move { (i, node.get(req).await) }
        });
        let outcomes = futures::future::join_all(futures).await;

        // Gather each replica's copy (if any) and the set that responded.
        let mut copies: Vec<(usize, proto::KnowledgeObject)> = Vec::new();
        let mut responded: Vec<usize> = Vec::new();
        for (i, outcome) in outcomes {
            match outcome {
                Ok(resp) => {
                    self.set_health(i, true);
                    responded.push(i);
                    if let Some(obj) = resp.object {
                        copies.push((i, obj));
                    }
                }
                Err(_) => self.set_health(i, false),
            }
        }

        // Last-write-wins: the freshest copy by `updated_at` (RFC3339 sorts
        // lexicographically; an empty timestamp is treated as oldest).
        let winner = copies
            .iter()
            .max_by(|a, b| a.1.updated_at.cmp(&b.1.updated_at))
            .map(|(_, o)| o.clone());

        if let Some(obj) = &winner {
            // Repair any responding replica whose copy is missing, staler, or
            // divergent in content from the winner.
            let stale: Vec<usize> = responded
                .iter()
                .copied()
                .filter(|i| match copies.iter().find(|(ci, _)| ci == i) {
                    Some((_, o)) => o.updated_at < obj.updated_at || o.content != obj.content,
                    None => true,
                })
                .collect();
            if !stale.is_empty() {
                let repair = knowledge_to_insert_req(obj);
                for i in stale {
                    if self.nodes[i].insert(repair.clone()).await.is_ok() {
                        debug!("read-repair: reconciled {} on node {i}", obj.id);
                    }
                }
            }
        }
        Ok(proto::GetResponse { object: winner })
    }

    pub async fn search(&self, req: proto::SearchRequest) -> Result<proto::SearchResponse, Status> {
        let top_k = req.top_k.max(1) as usize;
        let responses = self
            .scatter(|n| {
                let r = req.clone();
                async move { n.search(r).await }
            })
            .await?;

        let mut by_id: HashMap<String, proto::SearchResult> = HashMap::new();
        for resp in responses {
            for r in resp.results {
                by_id
                    .entry(r.id.clone())
                    .and_modify(|e| {
                        if r.score > e.score {
                            *e = r.clone();
                        }
                    })
                    .or_insert(r);
            }
        }
        let mut merged: Vec<_> = by_id.into_values().collect();
        merged.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        merged.truncate(top_k);

        Ok(proto::SearchResponse {
            total_found: merged.len() as i32,
            results: merged,
            plan_explanation: format!("distributed scatter-gather over {} nodes", self.nodes.len()),
        })
    }

    pub async fn ask(&self, req: proto::AskRequest) -> Result<proto::AskResponse, Status> {
        let max_results = req.max_results.max(1) as usize;
        let detected_intent = String::new();
        let responses = self
            .scatter(|n| {
                let r = req.clone();
                async move { n.ask(r).await }
            })
            .await?;

        let mut intent = detected_intent;
        let mut by_id: HashMap<String, proto::AskResult> = HashMap::new();
        for resp in responses {
            if intent.is_empty() {
                intent = resp.detected_intent;
            }
            for r in resp.results {
                by_id
                    .entry(r.id.clone())
                    .and_modify(|e| {
                        if r.relevance_score > e.relevance_score {
                            *e = r.clone();
                        }
                    })
                    .or_insert(r);
            }
        }
        let mut merged: Vec<_> = by_id.into_values().collect();
        merged.sort_by(|a, b| {
            b.relevance_score
                .partial_cmp(&a.relevance_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        merged.truncate(max_results);

        Ok(proto::AskResponse {
            results: merged,
            plan_explanation: format!("distributed scatter-gather over {} nodes", self.nodes.len()),
            detected_intent: intent,
        })
    }

    pub async fn sql(&self, req: proto::SqlRequest) -> Result<proto::SqlResponse, Status> {
        // Per-node SQL over each partition. A top-level scalar aggregate
        // (COUNT/SUM/MIN/MAX, no GROUP BY) is *combined* across shards into one
        // global row; everything else is concatenated. (AVG and GROUP BY aren't
        // mergeable from partials alone and are concatenated — see
        // `merge_sql_aggregate`.)
        let responses = self
            .scatter(|n| {
                let r = req.clone();
                async move { n.sql(r).await }
            })
            .await?;
        let rows = match merge_sql_aggregate(&req.query, &responses) {
            Some(merged) => merged,
            None => responses.into_iter().flat_map(|r| r.rows).collect(),
        };
        Ok(proto::SqlResponse { rows })
    }

    pub async fn list(&self, req: proto::ListRequest) -> Result<proto::ListResponse, Status> {
        let offset = req.offset as usize;
        let limit = if req.limit == 0 {
            100
        } else {
            req.limit as usize
        };
        // Over-fetch (offset+limit) from each node, merge by id, then page.
        let per_node = proto::ListRequest {
            offset: 0,
            limit: (offset + limit) as u32,
        };
        let responses = self
            .scatter(|n| {
                let r = per_node;
                async move { n.list(r).await }
            })
            .await?;

        let total: u64 = responses.iter().map(|r| r.total).sum();
        let mut by_id: HashMap<String, proto::KnowledgeObject> = HashMap::new();
        for resp in responses {
            for obj in resp.objects {
                by_id.entry(obj.id.clone()).or_insert(obj);
            }
        }
        let mut objects: Vec<_> = by_id.into_values().collect();
        objects.sort_by(|a, b| a.id.cmp(&b.id)); // stable global order
        let objects = objects.into_iter().skip(offset).take(limit).collect();
        Ok(proto::ListResponse { objects, total })
    }

    pub async fn stats(&self, req: proto::StatsRequest) -> Result<proto::StatsResponse, Status> {
        let responses = self
            .scatter(|n| {
                let r = req;
                async move { n.stats(r).await }
            })
            .await?;

        let mut out = proto::StatsResponse::default();
        let count = responses.len().max(1) as f64;
        let mut hit_rate_sum = 0.0;
        for resp in &responses {
            out.total_objects += resp.total_objects;
            out.vector_count += resp.vector_count;
            out.index_size_bytes += resp.index_size_bytes;
            out.graph_nodes += resp.graph_nodes;
            out.graph_edges += resp.graph_edges;
            out.active_agent_sessions += resp.active_agent_sessions;
            out.total_cost_usd += resp.total_cost_usd;
            out.cache_entries += resp.cache_entries;
            hit_rate_sum += resp.cache_hit_rate;
        }
        out.cache_hit_rate = hit_rate_sum / count;
        Ok(out)
    }

    pub async fn record_turn(
        &self,
        req: proto::RecordTurnRequest,
    ) -> Result<proto::RecordTurnResponse, Status> {
        let replicas = self.replicas_for_key(&req.session_id);
        let mut out = proto::RecordTurnResponse { turn_count: 0 };
        for &i in &replicas {
            if let Ok(resp) = self.nodes[i].record_turn(req.clone()).await {
                out = resp;
            }
        }
        Ok(out)
    }

    pub async fn get_session(
        &self,
        req: proto::GetSessionRequest,
    ) -> Result<proto::GetSessionResponse, Status> {
        for &i in &self.replicas_for_key(&req.session_id) {
            if let Ok(resp) = self.nodes[i].get_session(req.clone()).await {
                if resp.found {
                    return Ok(resp);
                }
            }
        }
        Ok(proto::GetSessionResponse {
            found: false,
            turns: Vec::new(),
        })
    }

    /// Run `f` against every **healthy** node in parallel, updating health from
    /// the outcomes. Tolerates partial failure (drops errored nodes), but errors
    /// if no healthy node responds — so callers can tell "no matches" (empty Ok)
    /// from "cluster unavailable" (Err).
    pub(crate) async fn scatter<F, Fut, T>(&self, f: F) -> Result<Vec<T>, Status>
    where
        F: Fn(Arc<dyn ClusterNode>) -> Fut,
        Fut: std::future::Future<Output = Result<T, Status>>,
    {
        let candidates: Vec<usize> = (0..self.nodes.len())
            .filter(|&i| self.is_healthy(i))
            .collect();
        if candidates.is_empty() && !self.nodes.is_empty() {
            return Err(Status::unavailable("no healthy cluster nodes"));
        }

        let futures = candidates.iter().map(|&i| {
            let fut = f(self.nodes[i].clone());
            async move { (i, fut.await) }
        });
        let outcomes = futures::future::join_all(futures).await;
        let attempted = outcomes.len();

        let mut oks = Vec::with_capacity(attempted);
        let mut last_err = None;
        for (i, outcome) in outcomes {
            match outcome {
                Ok(v) => {
                    self.set_health(i, true);
                    oks.push(v);
                }
                Err(e) => {
                    self.set_health(i, false);
                    last_err = Some(e);
                }
            }
        }
        if oks.is_empty() && attempted > 0 {
            return Err(last_err.unwrap_or_else(|| Status::unavailable("all cluster nodes failed")));
        }
        Ok(oks)
    }
}
