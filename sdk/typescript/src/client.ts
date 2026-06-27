/**
 * KowitoDB TypeScript SDK — gRPC client.
 *
 * Usage:
 *   import { KowitoDBClient } from "@kowitodb/sdk";
 *
 *   const db = new KowitoDBClient("localhost:50051");
 *   await db.connect();
 *   await db.remember("OpenAI raised $6.6B in 2024", {
 *     keywords: ["openai", "funding"],
 *     metadata: { company: "OpenAI" },
 *   });
 *   const res = await db.ask("Which companies raised funding?");
 *   for (const r of res.results) {
 *     console.log(`[${r.relevance_score.toFixed(2)}] ${r.content}`);
 *   }
 *   db.close();
 */

import { credentials, ChannelCredentials, ClientUnaryCall } from "@grpc/grpc-js";

import {
  loadKowitoDBService,
  KowitoDBGrpcClient,
  UnaryCallback,
} from "./service";
import type {
  AskResponse,
  ConversationTurnProto,
  DeleteRequest,
  DeleteResponse,
  GetRequest,
  GetResponse,
  GetSessionRequest,
  GetSessionResponse,
  InsertOptions,
  InsertRequest,
  InsertResponse,
  KnowledgeObject,
  RecordTurnRequest,
  RecordTurnResponse,
  RelationshipInput,
  RememberOptions,
  RememberRequest,
  RememberResponse,
  SearchRequest,
  SearchResponse,
  SearchResult,
  SqlRequest,
  SqlResponse,
  StatsResponse,
  UpdateOptions,
  UpdateRequest,
  UpdateResponse,
} from "./types";

export interface KowitoDBClientOptions {
  /** Channel credentials. Defaults to insecure (matching the Python SDK). */
  credentials?: ChannelCredentials;
}

const DEFAULT_ADDRESS = "localhost:50051";
const DEFAULT_IMPORTANCE = 0.5;

/**
 * Promisify a unary gRPC call.
 *
 * `invoke` receives a Node-style callback and is expected to kick off the RPC.
 * Driving the call through a caller-supplied closure (rather than passing a
 * bound method) keeps the generic `TResponse` precise — binding an overloaded
 * gRPC method would otherwise collapse its signature and erase the type.
 */
function callUnary<TResponse>(
  invoke: (callback: UnaryCallback<TResponse>) => ClientUnaryCall,
): Promise<TResponse> {
  return new Promise<TResponse>((resolve, reject) => {
    invoke((error, response) => {
      if (error) {
        reject(error);
        return;
      }
      // With proto-loader `defaults: true`, a successful unary call always
      // yields a response object.
      resolve(response as TResponse);
    });
  });
}

/**
 * gRPC client for KowitoDB.
 *
 * Mirrors the Python `KowitoDBClient` ergonomics: same high-level methods
 * (`remember`, `ask`, `forget`, `sql`) and low-level methods (`insert`,
 * `get`, `search`, `stats`) with an explicit `connect()` / `close()`
 * connection lifecycle. The connection is lazily established, so calling a
 * method without `connect()` works too.
 */
export class KowitoDBClient {
  readonly address: string;
  private readonly options: KowitoDBClientOptions;
  private stub: KowitoDBGrpcClient | undefined;

  constructor(address: string = DEFAULT_ADDRESS, options: KowitoDBClientOptions = {}) {
    this.address = address;
    this.options = options;
  }

  // ---- Connection ----

  /** Establish the gRPC connection. Idempotent. */
  connect(): void {
    if (this.stub) {
      return;
    }
    const ServiceClient = loadKowitoDBService();
    const creds = this.options.credentials ?? credentials.createInsecure();
    this.stub = new ServiceClient(this.address, creds);
  }

  /** Close the gRPC connection. */
  close(): void {
    if (this.stub) {
      this.stub.close();
      this.stub = undefined;
    }
  }

  private ensureConnected(): KowitoDBGrpcClient {
    if (!this.stub) {
      this.connect();
    }
    return this.stub as KowitoDBGrpcClient;
  }

  // ---- High-level AI API ----

  /**
   * ai.ask() — natural-language query with automatic retrieval.
   *
   * The engine detects intent, chooses retrieval strategies, searches all
   * indexes, reranks, and returns optimized results.
   */
  async ask(question: string, maxResults: number = 10): Promise<AskResponse> {
    const stub = this.ensureConnected();
    return callUnary<AskResponse>((cb) =>
      stub.ask({ question, max_results: maxResults }, cb),
    );
  }

  /**
   * ai.remember() — store knowledge for future retrieval.
   * Returns the new object ID.
   */
  async remember(content: string, options: RememberOptions = {}): Promise<string> {
    const stub = this.ensureConnected();
    const req: RememberRequest = {
      content,
      keywords: options.keywords ?? [],
      metadata: options.metadata ?? {},
      importance: options.importance ?? DEFAULT_IMPORTANCE,
    };
    const resp = await callUnary<RememberResponse>((cb) => stub.remember(req, cb));
    return resp.id;
  }

  /** Remove a knowledge object by ID. Returns whether it existed. */
  async forget(objectId: string): Promise<boolean> {
    const stub = this.ensureConnected();
    const req: DeleteRequest = { id: objectId };
    const resp = await callUnary<DeleteResponse>((cb) => stub.delete(req, cb));
    return resp.existed;
  }

  // ---- SQL API ----

  /**
   * Execute a SQL query against the DataFusion engine.
   *
   *   SELECT * FROM knowledge WHERE metadata.company = 'Acme'
   *   SELECT content FROM knowledge WHERE keyword LIKE '%enterprise%' LIMIT 10
   *
   * Returns one row per result; each row is a column-name -> value map.
   */
  async sql(query: string): Promise<Array<Record<string, string>>> {
    const stub = this.ensureConnected();
    const req: SqlRequest = { query };
    const resp = await callUnary<SqlResponse>((cb) => stub.sql(req, cb));
    return resp.rows.map((row) => row.columns);
  }

  // ---- Agent conversation memory ----

  /**
   * Append a turn to an agent conversation session, creating it if needed.
   * `role` is one of: user | assistant | system | observation.
   * Returns the total number of turns recorded in the session.
   */
  async recordTurn(
    sessionId: string,
    role: string,
    content: string,
  ): Promise<number> {
    const stub = this.ensureConnected();
    const req: RecordTurnRequest = {
      session_id: sessionId,
      role,
      content,
    };
    const resp = await callUnary<RecordTurnResponse>((cb) =>
      stub.recordTurn(req, cb),
    );
    return resp.turn_count;
  }

  /**
   * Fetch the recorded turns for a session, or null if the session does not
   * exist.
   */
  async getSession(
    sessionId: string,
  ): Promise<ConversationTurnProto[] | null> {
    const stub = this.ensureConnected();
    const req: GetSessionRequest = { session_id: sessionId };
    const resp = await callUnary<GetSessionResponse>((cb) =>
      stub.getSession(req, cb),
    );
    return resp.found ? resp.turns : null;
  }

  // ---- Low-level API ----

  /** Insert a knowledge object explicitly. Returns the new object ID. */
  async insert(content: string, options: InsertOptions = {}): Promise<string> {
    const stub = this.ensureConnected();
    const relationships: RelationshipInput[] = (options.relationships ?? []).map(
      ([relationType, targetId]) => ({
        relation_type: relationType,
        target_id: targetId,
      }),
    );
    const req: InsertRequest = {
      content,
      keywords: options.keywords ?? [],
      metadata: options.metadata ?? {},
      relationships,
      importance: options.importance ?? DEFAULT_IMPORTANCE,
    };
    const resp = await callUnary<InsertResponse>((cb) => stub.insert(req, cb));
    return resp.id;
  }

  /** Retrieve a knowledge object by ID, or null if it does not exist. */
  async get(objectId: string): Promise<KnowledgeObject | null> {
    const stub = this.ensureConnected();
    const req: GetRequest = { id: objectId };
    const resp = await callUnary<GetResponse>((cb) => stub.get(req, cb));
    return resp.object ?? null;
  }

  /**
   * Update an existing knowledge object. Only the provided fields are changed:
   * `content` (if set) is replaced and re-embedded, `metadata` is merged,
   * non-empty `keywords` replace the existing list, and `importance` is set.
   * Returns the full update response (whether it applied and the new version).
   */
  async update(objectId: string, options: UpdateOptions = {}): Promise<UpdateResponse> {
    const stub = this.ensureConnected();
    const req: UpdateRequest = {
      id: objectId,
      content: options.content,
      metadata: options.metadata ?? {},
      keywords: options.keywords ?? [],
      importance: options.importance,
      change_description: options.changeDescription,
    };
    return callUnary<UpdateResponse>((cb) => stub.update(req, cb));
  }

  /** Direct search (bypasses the AI planner). */
  async search(query: string, topK: number = 20): Promise<SearchResult[]> {
    const stub = this.ensureConnected();
    const req: SearchRequest = { query, top_k: topK };
    const resp = await callUnary<SearchResponse>((cb) => stub.search(req, cb));
    return resp.results;
  }

  /** Return database statistics. */
  async stats(): Promise<StatsResponse> {
    const stub = this.ensureConnected();
    return callUnary<StatsResponse>((cb) => stub.stats({}, cb));
  }
}
