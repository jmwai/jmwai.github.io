# Building Cart Pilot: A Multi-Agent AI Shopping Assistant with Google ADK and A2A Protocol

*How we built a conversational e-commerce platform using Google's Agent Development Kit, Agent-to-Agent Protocol, and Google Cloud Platform*

---

## Introduction

Cart Pilot is an AI-powered shopping assistant that transforms the traditional e-commerce experience into a natural conversation. Instead of clicking through menus and forms, users simply tell the AI what they want, and intelligent agents orchestrate the entire shopping journey—from product discovery to order placement.

This blog post dives deep into how we built Cart Pilot, exploring the architecture, agent system, Google Cloud technologies, and the protocols that make it all work together seamlessly.

---

## The Vision: Conversational Commerce

Traditional e-commerce platforms require users to navigate complex interfaces, fill out forms, and understand intricate workflows. Cart Pilot flips this model on its head by introducing **agent-driven architecture**, where users interact naturally through conversation, and AI agents handle all the complexity behind the scenes.

### Key Innovation: Multi-Agent Orchestration

Rather than a single monolithic AI, Cart Pilot uses a **hierarchical agent system**:
- **Shopping Agent** (Orchestrator): Understands user intent and routes to specialists
- **Product Discovery Agent**: Handles semantic and visual product search
- **Cart Agent**: Manages shopping cart operations
- **Checkout Agent**: Processes orders
- **Payment Agent**: Handles AP2-compliant payment processing
- **Customer Service Agent**: Provides support and handles returns

This architecture allows each agent to be an expert in its domain while maintaining a cohesive user experience.

---

## Architecture Overview

### High-Level System Design

```
┌─────────────────────────────────────────────────────────────┐
│                    Frontend (Next.js)                        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │  Chatbox UI  │  │ Product Grid │  │ Product Pages │     │
│  └──────┬───────┘  └──────────────┘  └──────────────┘     │
│         │                                                   │
│         │ A2A Protocol (JSON-RPC 2.0 over HTTP)             │
│         │                                                   │
└─────────┼───────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────┐
│                    Backend (FastAPI + ADK)                   │
│                                                               │
│  ┌──────────────────────────────────────────────────────┐  │
│  │         A2A Executor (ShoppingAgentExecutor)          │  │
│  │  - Receives A2A requests                              │  │
│  │  - Manages session state                              │  │
│  │  - Streams responses                                  │  │
│  │  - Creates artifacts (products, cart, orders)        │  │
│  └────────────────┬─────────────────────────────────────┘  │
│                   │                                         │
│                   ▼                                         │
│  ┌──────────────────────────────────────────────────────┐  │
│  │          Shopping Agent (Orchestrator)                │  │
│  │  - Routes requests to sub-agents                     │  │
│  │  - Maintains conversation context                     │  │
│  │  - Uses LLM-driven delegation                        │  │
│  └──┬──────────┬──────────┬──────────┬───────────────┘  │
│     │          │          │          │                   │
│     ▼          ▼          ▼          ▼                   │
│  ┌──────┐  ┌──────┐  ┌──────┐  ┌──────────────┐        │
│  │ Cart │  │Check │  │Product│  │  Customer    │        │
│  │Agent │  │out   │  │Discover│  │  Service    │        │
│  │      │  │Agent │  │  Agent │  │  Agent      │        │
│  └──────┘  └──────┘  └──────┘  └──────────────┘        │
│                                                           │
│  ┌────────────────────────────────────────────────────┐  │
│  │         Payment Agent (Separate)                    │  │
│  │  - AP2-compliant payment processing                 │  │
│  └────────────────────────────────────────────────────┘  │
│                                                           │
│  ┌────────────────────────────────────────────────────┐  │
│  │         Database (PostgreSQL + pgvector)           │  │
│  │  - Products, Cart, Orders, Payments                 │  │
│  └────────────────────────────────────────────────────┘  │
└───────────────────────────────────────────────────────────┘
```

---

## How Agents Work: The Multi-Agent System

### The Shopping Agent: Master Orchestrator

The Shopping Agent is the brain of the system. It doesn't perform actions directly; instead, it acts as an intelligent router that understands user intent and delegates to specialized sub-agents.

**Key Responsibilities:**
- **Intent Recognition**: Analyzes natural language to understand what the user wants
- **Agent Routing**: Transfers requests to the appropriate sub-agent
- **Context Management**: Maintains conversation flow across agent boundaries
- **Workflow Coordination**: Handles multi-step processes that span multiple agents

**Example Flow:**
```
User: "Find me running shoes"
↓
Shopping Agent analyzes intent → Recognizes product search
↓
Transfers to Product Discovery Agent
↓
Product Discovery Agent searches and returns results
↓
Shopping Agent presents results to user
```

### Sub-Agents: Domain Experts

Each sub-agent is a specialist in its domain:

#### 1. Product Discovery Agent
**Capabilities:**
- Semantic text search using pgvector embeddings
- Visual similarity search using image embeddings
- Stores search results in session state for later reference

**Tools:**
- `text_vector_search()`: Searches products using text queries
- `image_vector_search()`: Finds visually similar products from uploaded images

**Technology:** Uses Vertex AI Multimodal Embedding Model to generate embeddings, then performs vector similarity search in PostgreSQL with pgvector extension.

#### 2. Cart Agent
**Capabilities:**
- Adds products to cart by matching descriptions to search results
- Updates quantities and removes items
- Calculates totals and displays cart contents

**Intelligent Matching:**
When a user says "add the blue shoes," the Cart Agent:
1. Accesses `state["current_results"]` (from Product Discovery Agent)
2. Matches "blue shoes" to products using fuzzy matching
3. Creates cart item and stores in `state["cart"]`
4. Returns updated cart to frontend

#### 3. Checkout Agent
**Capabilities:**
- Validates cart before checkout
- Prepares order summaries
- Creates orders after payment processing
- Manages order status and cancellations

**Workflow:**
1. Validates cart contents
2. Prepares order summary with shipping address
3. Waits for user confirmation
4. After payment processing, creates order automatically

#### 4. Payment Agent
**Capabilities:**
- AP2-compliant payment processing
- Creates cryptographic mandates (Cart Mandate and Payment Mandate)
- Processes payments securely
- Handles payment method selection

**AP2 Compliance:**
The Payment Agent implements the Agent Payment Protocol (AP2) specification:
- **Cart Mandate**: Cryptographic proof of shopping intent
- **Payment Mandate**: Cryptographic proof of payment authorization
- **Payment Processing**: Secure transaction handling

#### 5. Customer Service Agent
**Capabilities:**
- Creates customer inquiries
- Processes returns and refunds
- Searches FAQ knowledge base
- Tracks inquiry status

### Agent Communication: Shared State

All agents share the same session state, allowing seamless handoffs:

```python
# Product Discovery Agent stores results
tool_context.state["current_results"] = products

# Cart Agent accesses results
current_results = tool_context.state.get("current_results", [])
product = find_product_in_results("blue shoes", current_results)

# Cart Agent stores cart
tool_context.state["cart"] = cart_items

# Checkout Agent accesses cart
cart = tool_context.state.get("cart", [])
```

This shared state enables agents to work together without explicit communication, creating a cohesive user experience.

---

## Google Cloud Technologies

### Cloud Run: Serverless Container Platform

Both frontend and backend run on **Google Cloud Run**, providing:

**Benefits:**
- **Auto-scaling**: Scales from 0 to multiple instances based on traffic
- **Pay-per-use**: Only pay for actual request processing time
- **Managed Infrastructure**: No server management required
- **Fast Deployments**: Deploy new versions in seconds

**Configuration:**
- **Backend Service**: `cart-pilot-backend`
  - Memory: 512Mi
  - CPU: 1 vCPU
  - Min instances: 1, Max instances: 3
  - Port: 8080
  - Timeout: 300 seconds

- **Frontend Service**: `cart-pilot-frontend`
  - Memory: 512Mi
  - CPU: 1 vCPU
  - Min instances: 1, Max instances: 3
  - Port: 8080

### Cloud SQL: Managed PostgreSQL Database

**Why Cloud SQL:**
- Fully managed PostgreSQL with automatic backups
- High availability and failover support
- Integrated with Cloud Run via Unix socket connections
- Supports pgvector extension for vector similarity search

**Database Schema:**
- **Catalog Items**: Products with pgvector embeddings for semantic search
- **Cart Items**: Shopping cart contents per session
- **Orders**: Order records with items and status
- **Payments**: Payment records with AP2 mandate references
- **Inquiries**: Customer service inquiries

**Connection Method:**
Cloud Run connects to Cloud SQL using Unix socket connections:
```
postgresql+psycopg2://user:password@/database?host=/cloudsql/connection-name
```

This provides secure, low-latency database access without exposing database ports.

### Vertex AI: LLM and Embeddings

**Gemini 2.5 Flash:**
- Used for all agent LLM inference
- Fast response times for conversational interactions
- Supports multimodal inputs (text and images)
- Cost-effective for high-volume usage

**Multimodal Embedding Model:**
- Generates embeddings for both text and images
- Enables semantic product search
- Powers visual similarity search for uploaded images
- Returns 768-dimensional vectors stored in pgvector

**Usage Pattern:**
```python
# Text embedding for semantic search
embedding = vertex_ai.generate_embeddings(text="running shoes")
# Store in database for similarity search

# Image embedding for visual search
embedding = vertex_ai.generate_embeddings(image=image_bytes)
# Find similar products using vector similarity
```

### Artifact Registry: Container Image Storage

**Purpose:**
- Stores Docker images for Cloud Run services
- Versioned image management
- Integrated with Cloud Run for seamless deployments

**Workflow:**
1. GitHub Actions builds Docker images
2. Images pushed to Artifact Registry with commit SHA tags
3. Cloud Run pulls images from Artifact Registry
4. Services updated with zero downtime

### Secret Manager: Secure Configuration

**Stored Secrets:**
- `google-api-key`: Vertex AI API key
- `db-password`: Database password
- `db-name`: Database name
- `db-user`: Database username
- `cloud-sql-connection-name`: Cloud SQL connection string

**Benefits:**
- Secrets never exposed in code or environment variables
- Automatic rotation support
- Fine-grained access control
- Audit logging

### Cloud Logging & Monitoring

**Automatic Logging:**
- All Cloud Run logs automatically captured
- Structured logging for easy querying
- Error reporting and alerting
- Request tracing across services

---

## Frontend-Backend Communication: A2A Protocol

### What is A2A Protocol?

The **Agent-to-Agent (A2A) Protocol** is a standardized communication protocol for AI agents, built on JSON-RPC 2.0. It enables:

- **Standardized Communication**: Consistent format for agent interactions
- **Streaming Support**: Real-time incremental updates
- **Structured Data Exchange**: Artifacts for complex data types
- **Session Continuity**: Maintains context across requests

### Communication Flow

```
1. User sends message in Chatbox
   ↓
2. Frontend: ShoppingAPI.sendMessageStream()
   - Wraps message in A2A format
   - Includes contextId for session continuity
   ↓
3. A2A Client sends POST to backend
   Endpoint: /.well-known/agent-card.json
   ↓
4. Backend: A2A Executor receives request
   - Extracts user message
   - Gets/creates session using contextId
   - Preserves existing state
   ↓
5. ADK Runner executes Shopping Agent
   - Shopping Agent routes to sub-agent
   - Sub-agent calls tools
   - Tools update session state
   ↓
6. Executor streams events
   - Text chunks (incremental)
   - Status updates
   - Artifacts (products, cart, orders)
   ↓
7. Frontend receives streaming events
   - Parses events using a2a-parser.ts
   - Updates UI incrementally
   - Displays artifacts in chatbox
```

### A2A Protocol Details

#### Agent Card

The Agent Card (`/.well-known/agent-card.json`) defines agent capabilities:

```json
{
  "name": "Shopping Assistant",
  "description": "AI-powered shopping assistant",
  "version": "1.0.0",
  "capabilities": {
    "streaming": true,
    "artifacts": true
  },
  "endpoints": {
    "request": "https://backend.run.app/a2a/request"
  }
}
```

The frontend uses this to initialize the A2A client and discover agent capabilities.

#### Request Format

```json
{
  "message": {
    "messageId": "uuid",
    "role": "user",
    "parts": [
      {
        "kind": "text",
        "text": "Find running shoes"
      }
    ],
    "kind": "message",
    "contextId": "session-context-id"
  },
  "configuration": {
    "blocking": false,
    "acceptedOutputModes": ["text/plain"]
  }
}
```

#### Response Format (Streaming)

The backend streams multiple events:

**Text Event:**
```json
{
  "id": 1,
  "jsonrpc": "2.0",
  "result": {
    "contextId": "session-context-id",
    "kind": "text-update",
    "text": "I found some running shoes..."
  }
}
```

**Artifact Event:**
```json
{
  "id": 2,
  "jsonrpc": "2.0",
  "result": {
    "contextId": "session-context-id",
    "kind": "artifact",
    "artifacts": [
      {
        "name": "products",
        "parts": [
          {
            "kind": "data",
            "data": {
              "type": "product_list",
              "products": [...]
            }
          }
        ]
      }
    ]
  }
}
```

**Status Event:**
```json
{
  "id": 3,
  "jsonrpc": "2.0",
  "result": {
    "contextId": "session-context-id",
    "kind": "status-update",
    "status": {
      "state": "working",
      "message": "Searching for products..."
    }
  }
}
```

#### Session Management

**Context ID:**
- Generated on first request
- Stored in frontend `localStorage`
- Sent with every request for session continuity
- Used as session ID in backend

**State Persistence:**
- Session state stored in PostgreSQL via `DatabaseSessionService`
- State persists across requests using `contextId`
- All agents share the same session state

---

## Google Agent Development Kit (ADK)

### What is ADK?

The **Agent Development Kit (ADK)** is Google's framework for building AI agents. It provides:

- **Agent Runtime**: Execution engine for agents
- **Session Management**: Persistent state across requests
- **Tool System**: Structured way to define agent capabilities
- **Memory Service**: Long-term memory for agents
- **Artifact Service**: Structured data exchange

### ADK Architecture

```
┌─────────────────────────────────────────────────────────┐
│              Agent Development Kit Runtime               │
│                                                          │
│  ┌──────────────────────────────────────────────────┐  │
│  │                    Runner                         │  │
│  │  - Event Processor                                │  │
│  │  - Manages agent execution                        │  │
│  └────────────┬─────────────────────────────────────┘  │
│               │                                         │
│               ▼                                         │
│  ┌──────────────────────────────────────────────────┐  │
│  │                Event Loop                         │  │
│  │  Ask ↔ Yield pattern for async execution          │  │
│  └────────────┬─────────────────────────────────────┘  │
│               │                                         │
│               ▼                                         │
│  ┌──────────────────────────────────────────────────┐  │
│  │            Execution Logic                        │  │
│  │  - Agent (Shopping Agent)                        │  │
│  │  - LLM Innovation (Gemini)                        │  │
│  │  - Callbacks                                       │  │
│  │  - Tools                                           │  │
│  └──────────────────────────────────────────────────┘  │
│                                                          │
│  ┌──────────────────────────────────────────────────┐  │
│  │                  Services                         │  │
│  │  - Session Service (Database-backed)              │  │
│  │  - Artifact Service (In-memory)                   │  │
│  │  - Memory Service (In-memory)                      │  │
│  └──────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────┘
```

### ADK Components

#### 1. Runner

The Runner is the core execution engine:

```python
from google.adk.runners import Runner

runner = Runner(
    app_name="shopping_agent",
    agent=shopping_agent,
    artifact_service=InMemoryArtifactService(),
    session_service=DatabaseSessionService(db_url=db_url),
    memory_service=InMemoryMemoryService(),
)
```

**Responsibilities:**
- Manages agent execution lifecycle
- Handles event loop (Ask/Yield pattern)
- Coordinates services (session, artifacts, memory)
- Streams events to executor

#### 2. Session Service

**DatabaseSessionService:**
- Stores session state in PostgreSQL
- Persists across application restarts
- Shared across all agents in a session
- Uses `contextId` as session identifier

**State Structure:**
```python
{
    "current_results": [...],  # Product search results
    "cart": [...],             # Cart items
    "current_order": {...},    # Current order
    "payment_data": {...}       # Payment information
}
```

#### 3. Agent Definition

Agents are defined using `LlmAgent`:

```python
from google.adk.agents import LlmAgent

root_agent = LlmAgent(
    name="shopping_agent",
    instruction="""You are the Shopping Agent...""",
    sub_agents=[
        product_discovery_agent,
        cart_agent,
        checkout_agent,
        customer_service_agent,
    ],
    planner=BuiltInPlanner(),
    model=types.GenerativeModel(
        model_name="gemini-2.5-flash",
        ...
    )
)
```

**Key Features:**
- **Sub-agents**: Hierarchical agent delegation
- **Planner**: Determines which tools/sub-agents to use
- **Model**: LLM for agent reasoning (Gemini 2.5 Flash)

#### 4. Tools

Tools are functions that agents can call:

```python
@tool
def add_to_cart(
    tool_context: ToolContext,
    product_description: str,
    quantity: int = 1
) -> Dict[str, Any]:
    # Access session state
    session_id = tool_context._invocation_context.session.id
    current_results = tool_context.state.get("current_results", [])
    
    # Find product
    product = find_product_in_results(product_description, current_results)
    
    # Create cart item
    cart_item = create_cart_item(product, quantity)
    
    # Update state
    tool_context.state["cart"] = cart_items
    
    return {"status": "success", "cart": cart_items}
```

**Tool Context Pattern:**
- All tools receive `ToolContext` as first parameter
- Provides access to session state
- Enables state sharing across agents

---

## Implementation Deep Dive

### Backend: A2A Executor

The `ShoppingAgentExecutor` bridges A2A protocol to ADK:

```python
class ShoppingAgentExecutor(AgentExecutor):
    def __init__(self):
        self.runner = Runner(
            app_name=self.agent.name,
            agent=self.agent,
            session_service=DatabaseSessionService(db_url=db_url),
            ...
        )
    
    async def execute(self, request_context: RequestContext):
        # Get or create session using contextId
        session = await self.runner.session_service.get_session(
            app_name=self.agent.name,
            user_id=user_id,
            session_id=request_context.task.context_id
        )
        
        # Execute agent
        async for event in self.runner.execute_stream(
            agent=self.agent,
            user_message=user_message,
            session=session
        ):
            # Stream events to frontend
            yield self._format_a2a_event(event)
```

**Key Responsibilities:**
- Receives A2A requests
- Manages session lifecycle
- Executes agents via ADK Runner
- Streams events in A2A format
- Creates artifacts from state changes

### Frontend: A2A Client Integration

The frontend uses the A2A Client SDK:

```typescript
import { A2AClient } from '@a2a-js/sdk/client';

class ShoppingAPI {
  private client: A2AClient | null = null;
  private contextId: string | null = null;
  
  async initialize(): Promise<void> {
    // Load agent card
    this.client = await A2AClient.fromCardUrl(AGENT_CARD_URL);
  }
  
  async *sendMessageStream(text?: string, image?: File) {
    // Create A2A message
    const message = {
      messageId: uuidv4(),
      role: 'user',
      parts: [
        text ? { kind: 'text', text } : null,
        image ? await this.createFilePart(image) : null
      ].filter(Boolean),
      contextId: this.contextId,
    };
    
    // Stream response
    for await (const event of this.client.sendMessageStream(message)) {
      // Extract contextId for session continuity
      if (event.result?.contextId) {
        this.contextId = event.result.contextId;
        localStorage.setItem('shopping_context_id', this.contextId);
      }
      
      yield event;
    }
  }
}
```

**Event Parsing:**
The frontend parses A2A events:

```typescript
export function parseStreamingEvent(event: any): StreamingEvent | null {
  if (event.result?.kind === 'text-update') {
    return {
      type: 'text',
      data: { text: event.result.text },
      isIncremental: true,
    };
  }
  
  if (event.result?.artifacts) {
    for (const artifact of event.result.artifacts) {
      if (artifact.name === 'products') {
        return {
          type: 'products',
          data: { products: extractProducts(artifact) },
          isIncremental: false,
        };
      }
      // ... handle other artifact types
    }
  }
  
  return null;
}
```

---

## Complete User Journey Example

Let's trace a complete shopping flow:

### 1. Product Search

**User:** "Find me running shoes"

**Flow:**
1. Frontend sends A2A request with message
2. Shopping Agent analyzes intent → recognizes product search
3. Transfers to Product Discovery Agent
4. Product Discovery Agent calls `text_vector_search("running shoes")`
5. Tool generates embedding using Vertex AI
6. Performs vector similarity search in PostgreSQL
7. Stores results in `state["current_results"]`
8. Returns products → Executor → Frontend
9. Frontend displays `ProductList` component

### 2. Add to Cart

**User:** "Add the blue ones"

**Flow:**
1. Shopping Agent recognizes cart operation
2. Transfers to Cart Agent
3. Cart Agent calls `add_to_cart(product_description="blue ones")`
4. Tool accesses `state["current_results"]`
5. Matches "blue ones" to product using fuzzy matching
6. Creates cart item in database
7. Updates `state["cart"]` with new cart items
8. Returns cart → Executor → Frontend
9. Frontend displays `CartDisplay` component

### 3. Checkout

**User:** "Checkout"

**Flow:**
1. Shopping Agent transfers to Checkout Agent
2. Checkout Agent calls `validate_cart_for_checkout()`
3. Tool validates cart contents
4. Checkout Agent calls `prepare_order_summary()`
5. Tool creates order summary from cart
6. Stores in `state["order_summary"]`
7. Returns order summary → Frontend
8. Frontend displays `OrderSummaryDisplay` component

### 4. Payment Processing

**User:** "Yes, confirm"

**Flow:**
1. Shopping Agent transfers to Payment Agent
2. Payment Agent calls `get_available_payment_methods()`
3. Returns payment methods → Frontend
4. User selects payment method
5. Payment Agent calls `create_cart_mandate()`
6. Creates AP2 Cart Mandate (cryptographic proof)
7. Payment Agent calls `create_payment_mandate()`
8. Creates AP2 Payment Mandate
9. Payment Agent calls `process_payment()`
10. Processes payment, stores in state
11. Returns "Payment processed successfully"
12. **Shopping Agent automatically transfers to Checkout Agent**
13. Checkout Agent calls `create_order()`
14. Creates order from cart and payment data
15. Clears cart
16. Returns order confirmation → Frontend
17. Frontend displays `OrderDisplay` component

---

## Deployment Architecture

### CI/CD Pipeline

**GitHub Actions Workflow:**

```yaml
on:
  push:
    branches: [main]
    paths:
      - 'backend/**'
      - 'frontend/**'

jobs:
  deploy-backend:
    if: contains(github.event.head_commit.modified, 'backend/')
    steps:
      - Build Docker image
      - Push to Artifact Registry
      - Deploy to Cloud Run
      
  deploy-frontend:
    if: contains(github.event.head_commit.modified, 'frontend/')
    steps:
      - Get backend service URL
      - Build Next.js with environment variables
      - Push to Artifact Registry
      - Deploy to Cloud Run
```

**Key Features:**
- Path-based triggers (only deploy changed services)
- Automatic image tagging with commit SHA
- Environment variable injection
- Zero-downtime deployments

### Security Architecture

**Secret Management:**
- All secrets stored in Secret Manager
- Injected at runtime (not in images)
- Fine-grained IAM policies
- Audit logging enabled

**Network Security:**
- Cloud SQL accessed via Unix sockets (no exposed ports)
- Cloud Run services communicate via HTTPS
- CORS configured for frontend-backend communication

**IAM Roles:**
- Service accounts for each component
- Least privilege principle
- Separate accounts for GitHub Actions and Cloud Run

---

## Key Technologies & Libraries

### Backend Stack

- **FastAPI**: Modern Python web framework
- **Google ADK**: Agent Development Kit for building AI agents
- **A2A Protocol**: Agent-to-Agent communication protocol
- **SQLAlchemy 2.0**: Modern ORM for database operations
- **pgvector**: PostgreSQL extension for vector similarity search
- **Vertex AI SDK**: Google Cloud AI services integration

### Frontend Stack

- **Next.js 16**: React framework with App Router
- **TypeScript**: Type-safe JavaScript
- **Tailwind CSS**: Utility-first CSS framework
- **A2A Client SDK**: JavaScript SDK for A2A protocol
- **React Hooks**: Custom hooks for state management

### Google Cloud Services

- **Cloud Run**: Serverless container platform
- **Cloud SQL**: Managed PostgreSQL database
- **Vertex AI**: LLM and embedding services
- **Artifact Registry**: Container image storage
- **Secret Manager**: Secure secret storage
- **Cloud Logging**: Centralized logging
- **Cloud Monitoring**: Metrics and alerting

---

## Performance Optimizations

### Database Optimizations

- **Connection Pooling**: Reuses database connections
- **Vector Indexing**: pgvector indexes for fast similarity search
- **Query Optimization**: Efficient queries with proper indexes

### Caching Strategy

- **Session State**: Cached in memory during request
- **Product Embeddings**: Pre-computed and stored in database
- **Agent Responses**: Streamed incrementally for perceived performance

### Scalability Features

- **Auto-scaling**: Cloud Run scales based on traffic
- **Stateless Design**: Services are stateless (state in database)
- **Horizontal Scaling**: Multiple instances handle concurrent requests

---

## Lessons Learned

### 1. Agent Design Patterns

**Hierarchical Delegation:**
- Orchestrator agents should delegate, not execute
- Sub-agents should be domain experts
- Shared state enables seamless handoffs

**State Management:**
- Centralized session state simplifies agent coordination
- State persistence enables multi-turn conversations
- State keys should be well-documented

### 2. Protocol Design

**A2A Protocol Benefits:**
- Standardized communication format
- Streaming support for real-time updates
- Artifacts for structured data exchange
- Session continuity via contextId

**Implementation Tips:**
- Always include contextId for session continuity
- Stream events incrementally for better UX
- Use artifacts for complex data structures

### 3. Google Cloud Best Practices

**Cloud Run:**
- Use appropriate memory/CPU allocations
- Set min instances for consistent performance
- Configure timeouts appropriately

**Cloud SQL:**
- Use Unix sockets for secure connections
- Enable connection pooling
- Monitor connection usage

**Secret Manager:**
- Never commit secrets to code
- Use service accounts with least privilege
- Rotate secrets regularly

---

## Future Enhancements

### Potential Improvements

1. **Enhanced Visual Search**: Better image matching algorithms
2. **Voice Interface**: Add voice input/output support
3. **Multi-language Support**: Internationalization
4. **Advanced Recommendations**: ML-based product recommendations
5. **Real-time Inventory**: Live inventory updates
6. **Social Features**: Share carts and wishlists

### Scalability Considerations

- **Caching Layer**: Redis for frequently accessed data
- **CDN Integration**: Cloud CDN for static assets
- **Load Balancing**: Multi-region deployment
- **Database Replication**: Read replicas for scaling reads

---

## Conclusion

Cart Pilot demonstrates the power of modern AI agent architecture combined with Google Cloud Platform. By leveraging:

- **Multi-agent orchestration** for specialized capabilities
- **A2A Protocol** for standardized communication
- **Google ADK** for agent runtime and management
- **Google Cloud** for scalable infrastructure

We've built a conversational e-commerce platform that feels natural and intuitive while handling complex workflows behind the scenes.

The combination of hierarchical agents, shared state, streaming responses, and cloud-native infrastructure creates a foundation for building sophisticated AI applications that can scale and evolve.

---

## Resources

- **Project Repository**: [GitHub Link]
- **Google ADK Documentation**: [ADK Docs]
- **A2A Protocol Specification**: [A2A Spec]
- **Google Cloud Run**: [Cloud Run Docs]
- **Vertex AI**: [Vertex AI Docs]

---

*Built with ❤️ using Google ADK, A2A Protocol, and Google Cloud Platform*

