//! `KowitoDBEngine` — graphrag methods (split from the former db.rs god-file).

use super::*;

impl KowitoDBEngine {
    /// Build (or rebuild) **GraphRAG community summaries**: detect communities in
    /// the knowledge graph (the auto-built `co_mentions` graph makes this work
    /// without explicit relationships), then summarize each community's members.
    /// With an `LlmClient` the summary is LLM-generated; otherwise it is a
    /// deterministic digest. Returns the number of communities summarized.
    ///
    /// This is the indexing half of Microsoft GraphRAG. It is **opt-in / on
    /// demand** precisely because full community summarization is token-expensive
    /// — call it after a batch ingest, not on every write.
    pub async fn build_community_summaries(&self) -> KResult<usize> {
        let communities = self.graph_index.detect_communities();
        let mut summaries = Vec::new();
        for members in communities
            .into_iter()
            .filter(|c| c.len() >= MIN_COMMUNITY_SIZE)
        {
            let mut texts = Vec::new();
            for &id in members.iter().take(COMMUNITY_SUMMARY_MAX_MEMBERS) {
                if let Some(obj) = self.get(id).await? {
                    texts.push(obj.content);
                }
            }
            if texts.is_empty() {
                continue;
            }
            let summary = self.summarize_community(&texts).await;
            summaries.push(CommunitySummary { members, summary });
        }
        let n = summaries.len();
        *self.community_summaries.lock() = summaries;
        info!("Built {n} GraphRAG community summaries");
        Ok(n)
    }

    /// Summarize one community's member texts (LLM when configured, else a
    /// deterministic digest).
    pub(crate) async fn summarize_community(&self, texts: &[String]) -> String {
        let joined = texts.join("\n- ");
        match &self.llm_client {
            Some(llm) => {
                let system = "Summarize these related items into a concise \
                    paragraph capturing the key entities, themes, and \
                    relationships among them. Output only the summary.";
                llm.complete(system, &joined).await.unwrap_or_else(|e| {
                    debug!("community summarization failed: {e}");
                    joined.chars().take(280).collect()
                })
            }
            None => format!(
                "Community of {} items: {}",
                texts.len(),
                joined.chars().take(280).collect::<String>()
            ),
        }
    }

    /// Answer a **global/holistic** question via GraphRAG map-reduce over the
    /// community summaries: each summary yields a partial answer (map), then the
    /// partials are combined into a final answer (reduce). Builds the summaries
    /// on first use. Returns `None` when no LLM client is configured or no
    /// community is relevant (callers fall back to ordinary retrieval).
    pub async fn global_query(&self, question: &str) -> KResult<Option<String>> {
        let Some(llm) = self.llm_client.clone() else {
            return Ok(None);
        };
        if self.community_summaries.lock().is_empty() {
            self.build_community_summaries().await?;
        }
        let summaries: Vec<String> = self
            .community_summaries
            .lock()
            .iter()
            .map(|c| c.summary.clone())
            .collect();
        if summaries.is_empty() {
            return Ok(None);
        }

        // Map: a partial answer from each community summary.
        let mut partials = Vec::new();
        for s in &summaries {
            let system = "Using ONLY the community summary, give a partial answer \
                to the question, or reply exactly NONE if the summary is \
                irrelevant.";
            let user = format!("Community summary:\n{s}\n\nQuestion: {question}");
            if let Ok(ans) = llm.complete(system, &user).await {
                let a = ans.trim();
                if !a.is_empty() && !a.eq_ignore_ascii_case("none") {
                    partials.push(a.to_string());
                }
            }
        }
        if partials.is_empty() {
            return Ok(None);
        }

        // Reduce: combine the partials into one comprehensive answer.
        let system = "Combine the partial answers into one comprehensive, \
            non-redundant final answer to the question. Output only the answer.";
        let user = format!(
            "Question: {question}\n\nPartial answers:\n- {}",
            partials.join("\n- ")
        );
        let final_answer = llm
            .complete(system, &user)
            .await
            .unwrap_or_else(|_| partials.join(" "));
        Ok(Some(final_answer))
    }

    /// Snapshot of the current community summaries (for inspection / RPC).
    pub fn community_summaries(&self) -> Vec<CommunitySummary> {
        self.community_summaries.lock().clone()
    }
}
