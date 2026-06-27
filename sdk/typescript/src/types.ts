/**
 * Typed message definitions for the KowitoDB gRPC API.
 *
 * These interfaces mirror the messages defined in `proto/kowitodb.proto`.
 * They describe the plain JavaScript objects produced/consumed by the
 * dynamically-loaded gRPC service (via @grpc/proto-loader).
 *
 * `proto-loader` is configured with `defaults: true`, so scalar fields are
 * always present on responses (with proto3 zero-values when unset). `map`
 * fields become plain objects, and `repeated` fields become arrays.
 */

// ---- Knowledge Object ----

export interface EmbeddingVector {
  values: number[];
}

export interface Relationship {
  relation_type: string;
  target_id: string;
  weight?: number;
}

export interface KnowledgeObject {
  id: string;
  content: string;
  embeddings: Record<string, EmbeddingVector>;
  metadata: Record<string, string>;
  keywords: string[];
  relationships: Relationship[];
  importance: number;
  created_at: string;
  updated_at: string;
}

// ---- Insert ----

export interface RelationshipInput {
  relation_type: string;
  target_id: string;
  weight?: number;
}

export interface InsertRequest {
  content: string;
  embeddings?: Record<string, EmbeddingVector>;
  metadata?: Record<string, string>;
  keywords?: string[];
  relationships?: RelationshipInput[];
  importance?: number;
}

export interface InsertResponse {
  id: string;
}

// ---- Batch insert ----

export interface BatchInsertRequest {
  items: InsertRequest[];
}

export interface BatchInsertResponse {
  ids: string[];
}

// ---- List / scroll ----

export interface ListRequest {
  offset: number;
  /** Page size; 0 means the server default. */
  limit: number;
}

export interface ListResponse {
  objects: KnowledgeObject[];
  /** Total number of objects in the store (for pagination). */
  total: number;
}

// ---- Get ----

export interface GetRequest {
  id: string;
}

export interface GetResponse {
  object?: KnowledgeObject;
}

// ---- Delete ----

export interface DeleteRequest {
  id: string;
}

export interface DeleteResponse {
  existed: boolean;
}

// ---- Search ----

export interface SearchRequest {
  query: string;
  top_k: number;
  /** Exact-match metadata constraints (ANDed); empty means no filtering. */
  metadata_filter?: Record<string, string>;
}

export interface SearchResult {
  id: string;
  content: string;
  score: number;
  metadata: Record<string, string>;
}

export interface SearchResponse {
  results: SearchResult[];
  plan_explanation: string;
  total_found: number;
}

// ---- ai.ask() ----

export interface AskRequest {
  question: string;
  max_results: number;
  max_context_tokens?: number;
  /** Exact-match metadata constraints (ANDed); empty means no filtering. */
  metadata_filter?: Record<string, string>;
}

export interface AskResult {
  id: string;
  content: string;
  relevance_score: number;
  metadata: Record<string, string>;
  /** vector, keyword, graph, metadata, time */
  retrieval_source: string;
}

export interface AskResponse {
  results: AskResult[];
  plan_explanation: string;
  detected_intent: string;
}

// ---- ai.remember() ----

export interface RememberRequest {
  content: string;
  embeddings?: Record<string, EmbeddingVector>;
  metadata?: Record<string, string>;
  keywords?: string[];
  importance?: number;
}

export interface RememberResponse {
  id: string;
}

// ---- Update ----

export interface UpdateRequest {
  id: string;
  /** If set, replaces content (and triggers re-embedding). */
  content?: string;
  /** Merged into existing metadata (keys overwrite). */
  metadata?: Record<string, string>;
  /** If non-empty, replaces keywords. */
  keywords?: string[];
  importance?: number;
  /** Recorded in the object's version history. */
  change_description?: string;
}

export interface UpdateResponse {
  updated: boolean;
  /** New length of the version history after this update. */
  version: number;
}

// ---- SQL ----

export interface SqlRequest {
  query: string;
}

export interface SqlRow {
  columns: Record<string, string>;
}

export interface SqlResponse {
  rows: SqlRow[];
}

// ---- Agent memory ----

export interface RecordTurnRequest {
  session_id: string;
  /** user | assistant | system | observation */
  role: string;
  content: string;
}

export interface RecordTurnResponse {
  turn_count: number;
}

export interface GetSessionRequest {
  session_id: string;
}

export interface ConversationTurnProto {
  role: string;
  content: string;
  timestamp: string;
}

export interface GetSessionResponse {
  found: boolean;
  turns: ConversationTurnProto[];
}

// ---- Stats ----

export type StatsRequest = Record<string, never>;

export interface StatsResponse {
  total_objects: number;
  vector_count: number;
  index_size_bytes: number;
  graph_nodes: number;
  graph_edges: number;
  active_agent_sessions: number;
  total_cost_usd: number;
  cache_entries: number;
  cache_hit_rate: number;
}

// ---- Optional argument bags for high-level client methods ----

export interface RememberOptions {
  keywords?: string[];
  metadata?: Record<string, string>;
  importance?: number;
}

export interface InsertOptions {
  keywords?: string[];
  metadata?: Record<string, string>;
  /** Tuples of [relation_type, target_id]. */
  relationships?: Array<[string, string]>;
  importance?: number;
}

export interface AskOptions {
  /** Maximum number of results to return. */
  maxResults?: number;
  /** Exact-match metadata constraints (ANDed); empty means no filtering. */
  metadataFilter?: Record<string, string>;
}

export interface SearchOptions {
  /** Maximum number of results to return. */
  topK?: number;
  /** Exact-match metadata constraints (ANDed); empty means no filtering. */
  metadataFilter?: Record<string, string>;
}

export interface UpdateOptions {
  /** If set, replaces content (and triggers re-embedding). */
  content?: string;
  /** Merged into existing metadata (keys overwrite). */
  metadata?: Record<string, string>;
  /** If non-empty, replaces keywords. */
  keywords?: string[];
  importance?: number;
  /** Recorded in the object's version history. */
  changeDescription?: string;
}
