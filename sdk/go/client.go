// Package kowitodb provides an idiomatic Go gRPC client for KowitoDB.
//
// Usage:
//
//	db, err := kowitodb.NewClient("localhost:50051")
//	if err != nil {
//		log.Fatal(err)
//	}
//	defer db.Close()
//
//	ctx := context.Background()
//	id, _ := db.Remember(ctx, "OpenAI raised $6.6B in 2024",
//		kowitodb.WithKeywords("openai", "funding"),
//		kowitodb.WithMetadata(map[string]string{"company": "OpenAI"}))
//
//	resp, _ := db.Ask(ctx, "Which companies raised funding?")
//	for _, r := range resp.Results {
//		fmt.Printf("[%.2f] %s\n", r.RelevanceScore, r.Content)
//	}
package kowitodb

import (
	"context"

	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"

	pb "github.com/kowito/kowitodb/sdk/go/kowitodbpb"
)

// DefaultAddress is used when NewClient is called with an empty address.
const DefaultAddress = "localhost:50051"

// Client is a high-level gRPC client for KowitoDB. It mirrors the ergonomics
// of the Python SDK's KowitoDBClient.
type Client struct {
	conn *grpc.ClientConn
	stub pb.KowitoDBClient
}

// NewClient dials a KowitoDB server and returns a connected Client.
// If addr is empty, DefaultAddress ("localhost:50051") is used.
// Additional grpc.DialOption values may be supplied to customise the
// connection (TLS, interceptors, etc.). By default an insecure connection
// is used.
func NewClient(addr string, opts ...grpc.DialOption) (*Client, error) {
	if addr == "" {
		addr = DefaultAddress
	}
	if len(opts) == 0 {
		opts = []grpc.DialOption{
			grpc.WithTransportCredentials(insecure.NewCredentials()),
		}
	}
	conn, err := grpc.NewClient(addr, opts...)
	if err != nil {
		return nil, err
	}
	return &Client{
		conn: conn,
		stub: pb.NewKowitoDBClient(conn),
	}, nil
}

// Close releases the underlying gRPC connection.
func (c *Client) Close() error {
	if c.conn == nil {
		return nil
	}
	return c.conn.Close()
}

// ---- Typed result models (mirroring the Python dataclasses) ----

// AskResult is a single result from Ask.
type AskResult struct {
	ID              string
	Content         string
	RelevanceScore  float32
	RetrievalSource string // vector, keyword, graph, metadata, time
	Metadata        map[string]string
}

func askResultFromProto(p *pb.AskResult) AskResult {
	return AskResult{
		ID:              p.GetId(),
		Content:         p.GetContent(),
		RelevanceScore:  p.GetRelevanceScore(),
		RetrievalSource: p.GetRetrievalSource(),
		Metadata:        p.GetMetadata(),
	}
}

// AskResponse is the response from Ask.
type AskResponse struct {
	Results         []AskResult
	PlanExplanation string
	DetectedIntent  string
}

// SearchResult is a single search result.
type SearchResult struct {
	ID       string
	Content  string
	Score    float32
	Metadata map[string]string
}

func searchResultFromProto(p *pb.SearchResult) SearchResult {
	return SearchResult{
		ID:       p.GetId(),
		Content:  p.GetContent(),
		Score:    p.GetScore(),
		Metadata: p.GetMetadata(),
	}
}

// KnowledgeObject is a stored knowledge object returned by Get and List.
type KnowledgeObject struct {
	ID         string
	Content    string
	Keywords   []string
	Metadata   map[string]string
	Importance float32
	CreatedAt  string
	UpdatedAt  string
}

func knowledgeObjectFromProto(o *pb.KnowledgeObject) *KnowledgeObject {
	if o == nil {
		return nil
	}
	return &KnowledgeObject{
		ID:         o.GetId(),
		Content:    o.GetContent(),
		Keywords:   o.GetKeywords(),
		Metadata:   o.GetMetadata(),
		Importance: o.GetImportance(),
		CreatedAt:  o.GetCreatedAt(),
		UpdatedAt:  o.GetUpdatedAt(),
	}
}

// Stats holds database statistics.
type Stats struct {
	TotalObjects        uint64
	VectorCount         uint64
	IndexSizeBytes      uint64
	GraphNodes          uint64
	GraphEdges          uint64
	ActiveAgentSessions uint64
	TotalCostUSD        float64
	CacheEntries        uint64
	CacheHitRate        float64
}

// ConversationTurn is a single recorded turn in an agent session.
type ConversationTurn struct {
	Role      string
	Content   string
	Timestamp string
}

// Relationship describes a directed relationship to another object.
type Relationship struct {
	RelationType string
	TargetID     string
}

// InsertItem describes a single object to store via BatchInsert. Only Content
// is required; the remaining fields are optional. Importance defaults to 0.5
// when left as the zero value.
type InsertItem struct {
	Content       string
	Keywords      []string
	Metadata      map[string]string
	Importance    float32
	Relationships []Relationship
}

func (it InsertItem) toProto() *pb.InsertRequest {
	importance := it.Importance
	if importance == 0 {
		importance = 0.5
	}
	rels := make([]*pb.RelationshipInput, 0, len(it.Relationships))
	for _, r := range it.Relationships {
		rels = append(rels, &pb.RelationshipInput{
			RelationType: r.RelationType,
			TargetId:     r.TargetID,
		})
	}
	return &pb.InsertRequest{
		Content:       it.Content,
		Keywords:      it.Keywords,
		Metadata:      it.Metadata,
		Relationships: rels,
		Importance:    importance,
	}
}

// ---- Options ----

// writeOptions collects the optional parameters shared by Remember/Insert.
type writeOptions struct {
	keywords      []string
	metadata      map[string]string
	importance    float32
	relationships []Relationship
}

// WriteOption configures Remember and Insert calls.
type WriteOption func(*writeOptions)

// WithKeywords attaches keywords to a stored object.
func WithKeywords(keywords ...string) WriteOption {
	return func(o *writeOptions) { o.keywords = keywords }
}

// WithMetadata attaches a metadata map to a stored object.
func WithMetadata(metadata map[string]string) WriteOption {
	return func(o *writeOptions) { o.metadata = metadata }
}

// WithImportance sets the importance score (default 0.5).
func WithImportance(importance float32) WriteOption {
	return func(o *writeOptions) { o.importance = importance }
}

// WithRelationships attaches relationships to an inserted object.
func WithRelationships(rels ...Relationship) WriteOption {
	return func(o *writeOptions) { o.relationships = rels }
}

func newWriteOptions(opts []WriteOption) writeOptions {
	o := writeOptions{importance: 0.5}
	for _, opt := range opts {
		opt(&o)
	}
	return o
}

// updateOptions collects the optional parameters for Update. Only fields that
// are explicitly set are sent; unset optional fields leave the stored object
// unchanged.
type updateOptions struct {
	content           *string
	metadata          map[string]string
	keywords          []string
	importance        *float32
	changeDescription *string
}

// UpdateOption configures an Update call.
type UpdateOption func(*updateOptions)

// WithUpdatedContent replaces the object's content (triggers re-embedding).
func WithUpdatedContent(content string) UpdateOption {
	return func(o *updateOptions) { o.content = &content }
}

// WithUpdatedMetadata merges the given keys into the object's metadata.
func WithUpdatedMetadata(metadata map[string]string) UpdateOption {
	return func(o *updateOptions) { o.metadata = metadata }
}

// WithUpdatedKeywords replaces the object's keywords. Passing a non-empty
// slice replaces the existing keywords.
func WithUpdatedKeywords(keywords ...string) UpdateOption {
	return func(o *updateOptions) { o.keywords = keywords }
}

// WithUpdatedImportance sets a new importance score.
func WithUpdatedImportance(importance float32) UpdateOption {
	return func(o *updateOptions) { o.importance = &importance }
}

// WithChangeDescription records a note in the object's version history.
func WithChangeDescription(description string) UpdateOption {
	return func(o *updateOptions) { o.changeDescription = &description }
}

func newUpdateOptions(opts []UpdateOption) updateOptions {
	var o updateOptions
	for _, opt := range opts {
		opt(&o)
	}
	return o
}

// queryOptions collects the optional parameters shared by Ask and Search.
type queryOptions struct {
	metadataFilter map[string]string
}

// QueryOption configures Ask and Search calls.
type QueryOption func(*queryOptions)

// WithMetadataFilter restricts results to objects whose metadata matches every
// given key/value pair (exact match, ANDed). An empty or nil map means no
// filtering.
func WithMetadataFilter(filter map[string]string) QueryOption {
	return func(o *queryOptions) { o.metadataFilter = filter }
}

func newQueryOptions(opts []QueryOption) queryOptions {
	var o queryOptions
	for _, opt := range opts {
		opt(&o)
	}
	return o
}

// ---- High-level AI API ----

// Remember stores knowledge for future retrieval (ai.remember()).
// It returns the new object's ID.
func (c *Client) Remember(ctx context.Context, content string, opts ...WriteOption) (string, error) {
	o := newWriteOptions(opts)
	resp, err := c.stub.Remember(ctx, &pb.RememberRequest{
		Content:    content,
		Keywords:   o.keywords,
		Metadata:   o.metadata,
		Importance: o.importance,
	})
	if err != nil {
		return "", err
	}
	return resp.GetId(), nil
}

// Ask runs a natural-language query with automatic retrieval (ai.ask()).
// The engine detects intent, chooses retrieval strategies, searches all
// indexes, reranks, and returns optimized results. If maxResults <= 0 a
// default of 10 is used. Pass WithMetadataFilter to restrict results to
// objects whose metadata matches the given key/value pairs.
func (c *Client) Ask(ctx context.Context, question string, maxResults int32, opts ...QueryOption) (*AskResponse, error) {
	if maxResults <= 0 {
		maxResults = 10
	}
	o := newQueryOptions(opts)
	resp, err := c.stub.Ask(ctx, &pb.AskRequest{
		Question:       question,
		MaxResults:     maxResults,
		MetadataFilter: o.metadataFilter,
	})
	if err != nil {
		return nil, err
	}
	results := make([]AskResult, 0, len(resp.GetResults()))
	for _, r := range resp.GetResults() {
		results = append(results, askResultFromProto(r))
	}
	return &AskResponse{
		Results:         results,
		PlanExplanation: resp.GetPlanExplanation(),
		DetectedIntent:  resp.GetDetectedIntent(),
	}, nil
}

// ---- Low-level API ----

// Insert stores a knowledge object explicitly and returns its ID.
func (c *Client) Insert(ctx context.Context, content string, opts ...WriteOption) (string, error) {
	o := newWriteOptions(opts)
	rels := make([]*pb.RelationshipInput, 0, len(o.relationships))
	for _, r := range o.relationships {
		rels = append(rels, &pb.RelationshipInput{
			RelationType: r.RelationType,
			TargetId:     r.TargetID,
		})
	}
	resp, err := c.stub.Insert(ctx, &pb.InsertRequest{
		Content:       content,
		Keywords:      o.keywords,
		Metadata:      o.metadata,
		Relationships: rels,
		Importance:    o.importance,
	})
	if err != nil {
		return "", err
	}
	return resp.GetId(), nil
}

// BatchInsert stores multiple knowledge objects in a single request and
// returns their IDs in the same order as the supplied items. Each item's
// Importance defaults to 0.5 when left as the zero value.
func (c *Client) BatchInsert(ctx context.Context, items []InsertItem) ([]string, error) {
	reqItems := make([]*pb.InsertRequest, 0, len(items))
	for _, it := range items {
		reqItems = append(reqItems, it.toProto())
	}
	resp, err := c.stub.BatchInsert(ctx, &pb.BatchInsertRequest{Items: reqItems})
	if err != nil {
		return nil, err
	}
	return resp.GetIds(), nil
}

// List returns a page of stored knowledge objects ordered by the server's
// default ordering, along with the total number of objects in the store (for
// pagination). A limit of 0 means the server default page size.
func (c *Client) List(ctx context.Context, offset, limit uint32) ([]KnowledgeObject, uint64, error) {
	resp, err := c.stub.List(ctx, &pb.ListRequest{Offset: offset, Limit: limit})
	if err != nil {
		return nil, 0, err
	}
	objects := make([]KnowledgeObject, 0, len(resp.GetObjects()))
	for _, o := range resp.GetObjects() {
		objects = append(objects, *knowledgeObjectFromProto(o))
	}
	return objects, resp.GetTotal(), nil
}

// Get retrieves a knowledge object by ID. It returns (nil, nil) when no
// object exists for the given ID.
func (c *Client) Get(ctx context.Context, id string) (*KnowledgeObject, error) {
	resp, err := c.stub.Get(ctx, &pb.GetRequest{Id: id})
	if err != nil {
		return nil, err
	}
	o := resp.GetObject()
	if o == nil {
		return nil, nil
	}
	return knowledgeObjectFromProto(o), nil
}

// Update modifies an existing knowledge object in place. Only the fields set
// via UpdateOption are changed; metadata keys are merged, while content,
// keywords, and importance replace their existing values. It returns whether
// the object was updated and the new length of its version history.
func (c *Client) Update(ctx context.Context, id string, opts ...UpdateOption) (updated bool, version uint32, err error) {
	o := newUpdateOptions(opts)
	resp, err := c.stub.Update(ctx, &pb.UpdateRequest{
		Id:                id,
		Content:           o.content,
		Metadata:          o.metadata,
		Keywords:          o.keywords,
		Importance:        o.importance,
		ChangeDescription: o.changeDescription,
	})
	if err != nil {
		return false, 0, err
	}
	return resp.GetUpdated(), resp.GetVersion(), nil
}

// Delete removes a knowledge object by ID and reports whether it existed.
func (c *Client) Delete(ctx context.Context, id string) (bool, error) {
	resp, err := c.stub.Delete(ctx, &pb.DeleteRequest{Id: id})
	if err != nil {
		return false, err
	}
	return resp.GetExisted(), nil
}

// Search performs a direct search, bypassing the AI planner. If topK <= 0 a
// default of 20 is used. Pass WithMetadataFilter to restrict results to
// objects whose metadata matches the given key/value pairs.
func (c *Client) Search(ctx context.Context, query string, topK int32, opts ...QueryOption) ([]SearchResult, error) {
	if topK <= 0 {
		topK = 20
	}
	o := newQueryOptions(opts)
	resp, err := c.stub.Search(ctx, &pb.SearchRequest{
		Query:          query,
		TopK:           topK,
		MetadataFilter: o.metadataFilter,
	})
	if err != nil {
		return nil, err
	}
	results := make([]SearchResult, 0, len(resp.GetResults()))
	for _, r := range resp.GetResults() {
		results = append(results, searchResultFromProto(r))
	}
	return results, nil
}

// Stats returns database statistics.
func (c *Client) Stats(ctx context.Context) (*Stats, error) {
	resp, err := c.stub.Stats(ctx, &pb.StatsRequest{})
	if err != nil {
		return nil, err
	}
	return &Stats{
		TotalObjects:        resp.GetTotalObjects(),
		VectorCount:         resp.GetVectorCount(),
		IndexSizeBytes:      resp.GetIndexSizeBytes(),
		GraphNodes:          resp.GetGraphNodes(),
		GraphEdges:          resp.GetGraphEdges(),
		ActiveAgentSessions: resp.GetActiveAgentSessions(),
		TotalCostUSD:        resp.GetTotalCostUsd(),
		CacheEntries:        resp.GetCacheEntries(),
		CacheHitRate:        resp.GetCacheHitRate(),
	}, nil
}

// Sql executes a SQL query against the DataFusion engine and returns the
// resulting rows. Each row is a map of column name to its string value.
func (c *Client) Sql(ctx context.Context, query string) ([]map[string]string, error) {
	resp, err := c.stub.Sql(ctx, &pb.SqlRequest{Query: query})
	if err != nil {
		return nil, err
	}
	rows := make([]map[string]string, 0, len(resp.GetRows()))
	for _, r := range resp.GetRows() {
		rows = append(rows, r.GetColumns())
	}
	return rows, nil
}

// ---- Agent conversation memory ----

// RecordTurn appends a turn to an agent conversation session, creating the
// session if it does not yet exist. The role is typically one of "user",
// "assistant", "system", or "observation". It returns the new turn count for
// the session.
func (c *Client) RecordTurn(ctx context.Context, sessionID, role, content string) (uint32, error) {
	resp, err := c.stub.RecordTurn(ctx, &pb.RecordTurnRequest{
		SessionId: sessionID,
		Role:      role,
		Content:   content,
	})
	if err != nil {
		return 0, err
	}
	return resp.GetTurnCount(), nil
}

// GetSession returns the recorded turns for an agent conversation session.
// It returns (nil, nil) when no session exists for the given ID.
func (c *Client) GetSession(ctx context.Context, sessionID string) ([]ConversationTurn, error) {
	resp, err := c.stub.GetSession(ctx, &pb.GetSessionRequest{SessionId: sessionID})
	if err != nil {
		return nil, err
	}
	if !resp.GetFound() {
		return nil, nil
	}
	turns := make([]ConversationTurn, 0, len(resp.GetTurns()))
	for _, t := range resp.GetTurns() {
		turns = append(turns, ConversationTurn{
			Role:      t.GetRole(),
			Content:   t.GetContent(),
			Timestamp: t.GetTimestamp(),
		})
	}
	return turns, nil
}
