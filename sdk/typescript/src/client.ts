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
 *
 * Authentication / deadlines / TLS:
 *
 *   const db = new KowitoDBClient("db.example.com:50051", {
 *     apiKey: process.env.KOWITODB_API_KEY, // `authorization: Bearer <key>`
 *     timeoutMs: 30_000,                    // default per-call deadline (0 = none)
 *     secure: true,                         // TLS with the system roots
 *   });
 */

import {
  credentials,
  ChannelCredentials,
  ClientUnaryCall,
  InterceptingCall,
  Interceptor,
} from "@grpc/grpc-js";

import {
  loadKowitoDBService,
  KowitoDBGrpcClient,
  UnaryCallback,
} from "./service";
import type {
  AskOptions,
  AskRequest,
  AskResponse,
  BatchInsertRequest,
  BatchInsertResponse,
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
  ListRequest,
  ListResponse,
  RecordTurnRequest,
  RecordTurnResponse,
  RelationshipInput,
  RememberOptions,
  RememberRequest,
  RememberResponse,
  SearchOptions,
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
  /**
   * Channel credentials. Defaults to insecure (matching the Python SDK), or
   * TLS with the system roots when `secure` is true.
   */
  credentials?: ChannelCredentials;
  /** Use TLS (`credentials.createSsl()`) when `credentials` is not given. */
  secure?: boolean;
  /**
   * API key sent as `authorization: Bearer <key>` metadata on every call
   * (must match the server's `--api-key` / `KOWITODB_API_KEY`). Attached via
   * an interceptor, so it works over insecure channels too (grpc-js call
   * credentials cannot be combined with insecure channel credentials).
   */
  apiKey?: string;
  /**
   * Default per-call deadline in milliseconds, applied to every call that has
   * no deadline of its own. Defaults to 30000; `0` disables it.
   */
  timeoutMs?: number;
}

const DEFAULT_ADDRESS = "localhost:50051";
const DEFAULT_IMPORTANCE = 0.5;
const DEFAULT_TIMEOUT_MS = 30_000;

/**
 * Build the interceptor that adds the API key and the default deadline to
 * every outgoing call.
 */
function makeInterceptor(apiKey: string | undefined, timeoutMs: number): Interceptor {
  return (options, nextCall) => {
    const callOptions =
      timeoutMs > 0 && options.deadline === undefined
        ? { ...options, deadline: new Date(Date.now() + timeoutMs) }
        : options;
    return new InterceptingCall(nextCall(callOptions), {
      start(metadata, listener, next) {
        if (apiKey && metadata.get("authorization").length === 0) {
          metadata.set("authorization", `Bearer ${apiKey}`);
        }
        next(metadata, listener);
      },
    });
  };
}

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
    const creds =
      this.options.credentials ??
      (this.options.secure ? credentials.createSsl() : credentials.createInsecure());
    const timeoutMs = this.options.timeoutMs ?? DEFAULT_TIMEOUT_MS;
    this.stub = new ServiceClient(this.address, creds, {
      interceptors: [makeInterceptor(this.options.apiKey, timeoutMs)],
    });
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
  async ask(
    question: string,
    maxResults: number = 10,
    options: AskOptions = {},
  ): Promise<AskResponse> {
    const stub = this.ensureConnected();
    const req: AskRequest = {
      question,
      max_results: options.maxResults ?? maxResults,
      metadata_filter: options.metadataFilter ?? {},
    };
    return callUnary<AskResponse>((cb) => stub.ask(req, cb));
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
    if (options.id) {
      req.id = options.id;
    }
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
   *   SELECT id, content FROM knowledge WHERE metadata LIKE '%"company":"Acme"%'
   *   SELECT content FROM knowledge WHERE keywords LIKE '%enterprise%' LIMIT 10
   *
   * `metadata` and `keywords` are JSON-encoded string columns.
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

  /** Build an InsertRequest from content + options (shared by insert/batchInsert). */
  private buildInsertRequest(content: string, options: InsertOptions): InsertRequest {
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
    if (options.id) {
      req.id = options.id;
    }
    return req;
  }

  /** Insert a knowledge object explicitly. Returns the new object ID. */
  async insert(content: string, options: InsertOptions = {}): Promise<string> {
    const stub = this.ensureConnected();
    const req = this.buildInsertRequest(content, options);
    const resp = await callUnary<InsertResponse>((cb) => stub.insert(req, cb));
    return resp.id;
  }

  /**
   * Insert multiple knowledge objects in a single call. Each item accepts the
   * same option bag as `insert`/`remember`. Returns the new object IDs in
   * request order.
   */
  async batchInsert(
    items: Array<{ content: string } & InsertOptions>,
  ): Promise<string[]> {
    const stub = this.ensureConnected();
    const req: BatchInsertRequest = {
      items: items.map(({ content, ...options }) =>
        this.buildInsertRequest(content, options),
      ),
    };
    const resp = await callUnary<BatchInsertResponse>((cb) =>
      stub.batchInsert(req, cb),
    );
    return resp.ids;
  }

  /**
   * List knowledge objects with pagination. Returns the page of objects and
   * the total number of objects in the store.
   */
  async list(
    offset: number = 0,
    limit: number = 0,
  ): Promise<{ objects: KnowledgeObject[]; total: number }> {
    const stub = this.ensureConnected();
    const req: ListRequest = { offset, limit };
    const resp = await callUnary<ListResponse>((cb) => stub.list(req, cb));
    return { objects: resp.objects, total: resp.total };
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
  async search(
    query: string,
    topK: number = 20,
    options: SearchOptions = {},
  ): Promise<SearchResult[]> {
    const stub = this.ensureConnected();
    const req: SearchRequest = {
      query,
      top_k: options.topK ?? topK,
      metadata_filter: options.metadataFilter ?? {},
    };
    const resp = await callUnary<SearchResponse>((cb) => stub.search(req, cb));
    return resp.results;
  }

  /** Return database statistics. */
  async stats(): Promise<StatsResponse> {
    const stub = this.ensureConnected();
    return callUnary<StatsResponse>((cb) => stub.stats({}, cb));
  }
}
