use std::collections::HashMap;
use std::sync::Arc;

use kowitodb_core::ObjectId;
use parking_lot::RwLock;

/// A single turn in an agent conversation.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ConversationTurn {
    /// The user's question or agent's observation.
    pub role: TurnRole,
    /// The text content.
    pub content: String,
    /// When this turn occurred.
    pub timestamp: chrono::DateTime<chrono::Utc>,
    /// IDs of knowledge objects referenced in this turn.
    pub referenced_objects: Vec<ObjectId>,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum TurnRole {
    User,
    Assistant,
    System,
    Observation,
}

/// An agent session — a persistent conversation with memory.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct AgentSession {
    /// Unique session ID.
    pub id: String,
    /// Conversation history.
    pub turns: Vec<ConversationTurn>,
    /// Working memory: key-value facts the agent has learned.
    pub working_memory: HashMap<String, String>,
    /// IDs of knowledge objects deemed important for this session.
    pub pinned_objects: Vec<ObjectId>,
    /// When the session was created.
    pub created_at: chrono::DateTime<chrono::Utc>,
    /// Last activity timestamp.
    pub last_active: chrono::DateTime<chrono::Utc>,
    /// Optional metadata (e.g., user ID, task context).
    pub metadata: HashMap<String, String>,
}

impl AgentSession {
    pub fn new(id: impl Into<String>) -> Self {
        let now = chrono::Utc::now();
        Self {
            id: id.into(),
            turns: Vec::new(),
            working_memory: HashMap::new(),
            pinned_objects: Vec::new(),
            created_at: now,
            last_active: now,
            metadata: HashMap::new(),
        }
    }

    /// Add a turn to the conversation.
    pub fn add_turn(&mut self, role: TurnRole, content: impl Into<String>) {
        self.turns.push(ConversationTurn {
            role,
            content: content.into(),
            timestamp: chrono::Utc::now(),
            referenced_objects: Vec::new(),
        });
        self.last_active = chrono::Utc::now();
    }

    /// Add a turn with referenced knowledge objects.
    pub fn add_turn_with_refs(
        &mut self,
        role: TurnRole,
        content: impl Into<String>,
        refs: Vec<ObjectId>,
    ) {
        self.turns.push(ConversationTurn {
            role,
            content: content.into(),
            timestamp: chrono::Utc::now(),
            referenced_objects: refs,
        });
        self.last_active = chrono::Utc::now();
    }

    /// Remember a key-value fact.
    pub fn remember_fact(&mut self, key: impl Into<String>, value: impl Into<String>) {
        self.working_memory.insert(key.into(), value.into());
    }

    /// Recall a remembered fact.
    pub fn recall_fact(&self, key: &str) -> Option<&String> {
        self.working_memory.get(key)
    }

    /// Pin a knowledge object as important for this session.
    pub fn pin_object(&mut self, id: ObjectId) {
        if !self.pinned_objects.contains(&id) {
            self.pinned_objects.push(id);
        }
    }

    /// Get the last N turns of conversation.
    pub fn recent_turns(&self, n: usize) -> &[ConversationTurn] {
        let start = self.turns.len().saturating_sub(n);
        &self.turns[start..]
    }

    /// Count turns in this session.
    pub fn turn_count(&self) -> usize {
        self.turns.len()
    }
}

/// Manages multiple agent sessions — `ai.remember()` at the session level.
pub struct AgentMemory {
    sessions: Arc<RwLock<HashMap<String, AgentSession>>>,
}

impl AgentMemory {
    pub fn new() -> Self {
        Self {
            sessions: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Create or retrieve a session.
    pub fn get_or_create(&self, session_id: impl Into<String>) -> AgentSession {
        let id = session_id.into();
        let mut sessions = self.sessions.write();
        sessions
            .entry(id.clone())
            .or_insert_with(|| AgentSession::new(id))
            .clone()
    }

    /// Save/update a session.
    pub fn save(&self, session: AgentSession) {
        let mut sessions = self.sessions.write();
        sessions.insert(session.id.clone(), session);
    }

    /// Get a session by ID.
    pub fn get(&self, session_id: &str) -> Option<AgentSession> {
        self.sessions.read().get(session_id).cloned()
    }

    /// Delete a session.
    pub fn delete(&self, session_id: &str) -> bool {
        self.sessions.write().remove(session_id).is_some()
    }

    /// List all active session IDs.
    pub fn list_sessions(&self) -> Vec<String> {
        self.sessions.read().keys().cloned().collect()
    }

    /// Count active sessions.
    pub fn session_count(&self) -> usize {
        self.sessions.read().len()
    }
}

impl Default for AgentMemory {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_agent_session_basics() {
        let mut session = AgentSession::new("test-session");
        session.add_turn(TurnRole::User, "What are enterprise customers?");
        session.add_turn(TurnRole::Assistant, "Enterprise customers are...");
        session.remember_fact("last_topic", "enterprise");

        assert_eq!(session.turn_count(), 2);
        assert_eq!(
            session.recall_fact("last_topic"),
            Some(&"enterprise".to_string())
        );

        let recent = session.recent_turns(1);
        assert_eq!(recent.len(), 1);
        assert_eq!(recent[0].role, TurnRole::Assistant);
    }

    #[test]
    fn test_agent_memory_sessions() {
        let memory = AgentMemory::new();
        let mut session = memory.get_or_create("agent-1");
        session.add_turn(TurnRole::User, "Hello");
        memory.save(session);

        assert_eq!(memory.session_count(), 1);
        assert_eq!(memory.list_sessions(), vec!["agent-1"]);

        let loaded = memory.get("agent-1").unwrap();
        assert_eq!(loaded.turn_count(), 1);
    }
}
