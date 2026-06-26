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

// ---- Stats ----

export type StatsRequest = Record<string, never>;

export interface StatsResponse {
  total_objects: number;
  vector_count: number;
  index_size_bytes: number;
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
