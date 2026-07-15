//! `KowitoDBEngine` — memory methods (split from the former db.rs god-file).

use super::*;

impl KowitoDBEngine {
    /// Mem0-style memory distillation: extract the single durable fact worth
    /// remembering from a raw conversation turn. `Some(fact)` to promote that
    /// fact, `None` to skip promotion (the LLM judged it not memory-worthy, or
    /// the call failed). Only consulted when an LLM client is configured.
    pub(crate) async fn distill_memory(&self, content: &str) -> Option<String> {
        let llm = self.llm_client.as_ref()?;
        let system = "Extract the single most important durable fact or \
            preference from the message that is worth remembering long-term. \
            Reply with just that fact as a concise statement, or exactly NOOP if \
            nothing is worth remembering.";
        match llm.complete(system, content).await {
            Ok(s) if s.trim().eq_ignore_ascii_case("noop") => None,
            Ok(s) if !s.trim().is_empty() => Some(s.trim().to_string()),
            _ => None,
        }
    }

    /// Record an agent conversation turn AND promote it into searchable,
    /// graph-able knowledge (Mem0-style episodic memory). Returns the session's
    /// turn count.
    ///
    /// The memory object has a stable id derived from `(session_id, content)`, so
    /// re-recording the same turn is idempotent (no duplicate memory). `system`
    /// turns are not promoted. This makes past conversation retrievable via
    /// `ai.ask()` and linkable in the graph alongside ingested knowledge.
    pub async fn remember_turn(
        &self,
        session_id: &str,
        role: &str,
        content: String,
    ) -> KResult<u32> {
        let turn_role = match role.to_lowercase().as_str() {
            "assistant" => TurnRole::Assistant,
            "system" => TurnRole::System,
            "observation" => TurnRole::Observation,
            _ => TurnRole::User,
        };

        let mut session = self.agent_memory.get_or_create(session_id);
        session.add_turn(turn_role.clone(), content.clone());
        let count = session.turn_count() as u32;
        self.agent_memory.save(session);

        // Promote to searchable knowledge (idempotent by stable id), linked in
        // the graph to the existing knowledge the turn mentions. With an LLM
        // client, the raw turn is first distilled to a salient durable fact
        // (Mem0-style); `None` means "nothing worth remembering" → skip.
        if !matches!(turn_role, TurnRole::System) && !content.trim().is_empty() {
            let promote = if self.llm_client.is_some() {
                self.distill_memory(&content).await
            } else {
                Some(content.clone())
            };
            if let Some(fact) = promote {
                let mem_id = stable_memory_id(session_id, &fact);
                if self.get(mem_id).await?.is_none() {
                    let related = self.find_related_objects(&fact, 3);
                    let mut obj = KnowledgeObject::new(fact)
                        .with_metadata("session_id", session_id)
                        .with_metadata("role", role)
                        .with_metadata("kind", "memory");
                    obj.id = mem_id;
                    for target in related {
                        obj = obj.with_relationship("mentions", target);
                    }
                    self.insert(obj).await?;
                }
            }
        }
        Ok(count)
    }
}
