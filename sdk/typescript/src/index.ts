/**
 * @kowitodb/sdk — TypeScript gRPC client for KowitoDB,
 * the AI Knowledge Operating System.
 */

export { KowitoDBClient } from "./client";
export type { KowitoDBClientOptions } from "./client";

export {
  loadKowitoDBService,
} from "./service";
export type {
  KowitoDBGrpcClient,
  KowitoDBGrpcClientConstructor,
  UnaryCallback,
} from "./service";

export type {
  // Knowledge object
  EmbeddingVector,
  Relationship,
  KnowledgeObject,
  // Insert
  RelationshipInput,
  InsertRequest,
  InsertResponse,
  InsertOptions,
  // Batch insert
  BatchInsertRequest,
  BatchInsertResponse,
  // List
  ListRequest,
  ListResponse,
  // Get
  GetRequest,
  GetResponse,
  // Update
  UpdateRequest,
  UpdateResponse,
  UpdateOptions,
  // Delete
  DeleteRequest,
  DeleteResponse,
  // Search
  SearchRequest,
  SearchResult,
  SearchResponse,
  SearchOptions,
  // Ask
  AskRequest,
  AskResult,
  AskResponse,
  AskOptions,
  // Remember
  RememberRequest,
  RememberResponse,
  RememberOptions,
  // SQL
  SqlRequest,
  SqlRow,
  SqlResponse,
  // Agent memory
  RecordTurnRequest,
  RecordTurnResponse,
  GetSessionRequest,
  ConversationTurnProto,
  GetSessionResponse,
  // Stats
  StatsRequest,
  StatsResponse,
} from "./types";
