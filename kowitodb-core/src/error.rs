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

/// Convenience result type.
pub type Result<T> = std::result::Result<T, KowitoError>;
