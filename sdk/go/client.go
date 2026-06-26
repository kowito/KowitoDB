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

// KnowledgeObject is a stored knowledge object returned by Get.
type KnowledgeObject struct {
	ID         string
	Content    string
	Keywords   []string
	Metadata   map[string]string
	Importance float32
	CreatedAt  string
	UpdatedAt  string
}

// Stats holds database statistics.
type Stats struct {
	TotalObjects   uint64
	VectorCount    uint64
	IndexSizeBytes uint64
}

// Relationship describes a directed relationship to another object.
type Relationship struct {
	RelationType string
	TargetID     string
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
// default of 10 is used.
func (c *Client) Ask(ctx context.Context, question string, maxResults int32) (*AskResponse, error) {
	if maxResults <= 0 {
		maxResults = 10
	}
	resp, err := c.stub.Ask(ctx, &pb.AskRequest{
		Question:   question,
		MaxResults: maxResults,
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
	return &KnowledgeObject{
		ID:         o.GetId(),
		Content:    o.GetContent(),
		Keywords:   o.GetKeywords(),
		Metadata:   o.GetMetadata(),
		Importance: o.GetImportance(),
		CreatedAt:  o.GetCreatedAt(),
		UpdatedAt:  o.GetUpdatedAt(),
	}, nil
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
// default of 20 is used.
func (c *Client) Search(ctx context.Context, query string, topK int32) ([]SearchResult, error) {
	if topK <= 0 {
		topK = 20
	}
	resp, err := c.stub.Search(ctx, &pb.SearchRequest{
		Query: query,
		TopK:  topK,
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
		TotalObjects:   resp.GetTotalObjects(),
		VectorCount:    resp.GetVectorCount(),
		IndexSizeBytes: resp.GetIndexSizeBytes(),
	}, nil
}
