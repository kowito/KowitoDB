use thiserror::Error;

/// Primary error type for KowitoDB.
#[derive(Error, Debug)]
pub enum KowitoError {
    #[error("Storage error: {0}")]
    Storage(String),

    #[error("Index error: {0}")]
    Index(String),

    #[error("Planning error: {0}")]
    Planner(String),

    #[error("Not found: {0}")]
    NotFound(String),

    #[error("Already exists: {0}")]
    AlreadyExists(String),

    #[error("Invalid input: {0}")]
    InvalidInput(String),

    #[error("Serialization error: {0}")]
    Serialization(String),

    #[error("Internal error: {0}")]
    Internal(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

impl KowitoError {
    /// A human-readable code string suitable for logging and client-side
    /// dispatching. Mirrors the gRPC status code intent without pulling in tonic.
    pub fn code(&self) -> &'static str {
        match self {
            KowitoError::NotFound(_) => "NOT_FOUND",
            KowitoError::AlreadyExists(_) => "ALREADY_EXISTS",
            KowitoError::InvalidInput(_) => "INVALID_ARGUMENT",
            KowitoError::Storage(_)
            | KowitoError::Index(_)
            | KowitoError::Planner(_)
            | KowitoError::Serialization(_)
            | KowitoError::Internal(_)
            | KowitoError::Io(_) => "INTERNAL",
        }
    }
}

/// Convenience result type.
pub type Result<T> = std::result::Result<T, KowitoError>;
