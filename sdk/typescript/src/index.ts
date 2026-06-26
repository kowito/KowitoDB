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
  // Get
  GetRequest,
  GetResponse,
  // Delete
  DeleteRequest,
  DeleteResponse,
  // Search
  SearchRequest,
  SearchResult,
  SearchResponse,
  // Ask
  AskRequest,
  AskResult,
  AskResponse,
  // Remember
  RememberRequest,
  RememberResponse,
  RememberOptions,
  // Stats
  StatsRequest,
  StatsResponse,
} from "./types";
