# FusionAI

FusionAI is a full-stack AI learning and research platform that combines an AI notebook, a personalized roadmap generator, and a live AI intelligence feed into one authenticated workspace. It is built around FastAPI, React, Supabase, Zep Cloud, ChromaDB, Redis, Groq, Gemini, Tavily, and custom Graph-RAG pipelines.

The project is designed as more than a chat wrapper: it ingests multiple source types, builds searchable learning memory, generates personalized study paths, tracks source context, retrieves graph evidence, synthesizes answers, and evaluates research answers against a live search baseline.

## Core Features

- **Fusion Notebook**: Upload and query documents, YouTube videos, web pages, audio files, and raw text inside authenticated notebooks.
- **Pathfinder AI**: Generate dependency-aware learning roadmaps for any topic and open individual roadmap nodes to generate educational content.
- **AI News Intelligence Feed**: Collect AI news, GitHub repositories, and arXiv papers, then run Graph-RAG style research queries over the collected intelligence.
- **Advanced Research Pipeline**: Query expansion, embedding routing, hybrid ranking, subtree retrieval, relevance gating, agentic orchestration, recursive analysis, final answer synthesis, and automatic benchmarking.
- **Memory and Personalization**: Uses Supabase authentication/storage and Zep Cloud notebook memory for persistent user-aware learning sessions.
- **Self-hosting/Fine-tuning Foundations**: Includes PEFT/LoRA and `ctransformers` tooling for experimentation with local or quantized models.

## Project Components

### 1. Fusion Notebook

The notebook component lets users create authenticated notebook sessions and attach multiple source types:

- PDF/document files
- YouTube videos
- Web pages
- Audio files
- Text notes

The backend extracts content, stores source metadata, indexes content with vector search, and supports chat-style question answering over notebook context. Zep Cloud is used for notebook conversational memory.

Relevant backend routes:

```text
backend/routes/document.py
backend/routes/youtube_video.py
backend/routes/web.py
backend/routes/audio.py
backend/routes/text_content.py
backend/routes/Notebook_Routes/
```

Frontend:

```text
frontend/src/app/components/NotebookSessions.tsx
frontend/src/app/components/MainLayout.tsx
frontend/src/app/components/SourcesSidebar.tsx
frontend/src/app/components/ChatMessage.tsx
frontend/src/app/components/ChatInput.tsx
```

### 2. Pathfinder AI

Pathfinder AI generates personalized learning roadmaps. A user enters a topic and difficulty level, and the backend produces a structured roadmap with nodes, dependencies, positions, status, and generated educational content.

Capabilities:

- AI-generated roadmap with 8-12 structured nodes
- Dependency-aware locked/unlocked node states
- Supabase persistence for roadmaps and nodes
- Lazy generation of node-level educational content
- Node completion and automatic unlocking of dependent topics

Relevant backend:

```text
backend/routes/Roadmap_Routes/roadmap_response.py
services/roadmap_service/
```

Frontend:

```text
frontend/src/app/components/RoadmapPage.tsx
frontend/src/app/components/RoadmapViewPage.tsx
frontend/src/app/components/NodeContentModal.tsx
```

### 3. AI News Research Pipeline

The AI News component collects live AI intelligence and provides an advanced research-query pipeline over the collected data.

Sources:

- GitHub repositories
- arXiv AI papers
- AI news articles

Core pipeline:

```text
User query
-> Query expansion
-> Embedding routing
-> Parent selection
-> Subtree fetch
-> Relevance gate
-> Optional live web evidence reinforcement
-> Agentic orchestration
-> Feature building
-> Gemini audit
-> Groq recursive analysis
-> Groq final answer synthesis
-> User-facing response
-> Tavily-vs-FusionAI benchmark evaluation
```

Relevant backend:

```text
backend/routes/Research_Routes/get_Latest_Data.py
backend/routes/Research_Routes/processor.py
backend/routes/Research_Routes/Nexus_Graph_DB/
backend/routes/Research_Routes/Query_Pipeline/
backend/routes/Research_Routes/benchmark_service.py
```

Frontend:

```text
frontend/src/app/components/AINewsPage.tsx
```

Detailed paper-style documentation:

```text
docs/research_pipeline_ai_news_paper.md
```

## Architecture

```text
frontend/
  React + TypeScript + Vite app
  Authenticated dashboard, notebook UI, roadmap UI, AI news UI

backend/
  FastAPI app
  Supabase auth middleware
  Notebook, roadmap, document, YouTube, web, audio, text, and AI news routes

services/
  Notebook/research/vector generation services
  ChromaDB-backed retrieval utilities
  Roadmap generation utilities

data/
  Benchmark and generated dataset artifacts

docs/
  Research pipeline documentation
```

## Tech Stack

### Frontend

- React
- TypeScript
- Vite
- Tailwind CSS
- Radix UI components
- Lucide React icons
- Supabase JS client
- React Router

### Backend

- Python
- FastAPI
- Uvicorn
- Supabase Python client
- Redis
- ChromaDB
- Zep Cloud
- Groq
- Google Gemini
- Tavily
- CrewAI
- LangChain Google GenAI
- Sentence Transformers
- PyMuPDF
- yt-dlp / YouTube transcript tooling
- AssemblyAI
- Firecrawl

### AI / Retrieval / Research

- Graph-RAG style storage and retrieval
- Hybrid ranking using semantic, lexical, keyword, entity, freshness, and source signals
- Query expansion
- Recursive analysis
- Agentic orchestration services
- Evaluation with LLM-as-judge
- PEFT/LoRA experimentation support
- `ctransformers` support for quantized local model experiments

## Repository Structure

```text
FusionAI/
  backend/
    app.py
    routes/
      Notebook_Routes/
      Roadmap_Routes/
      Research_Routes/
      document.py
      youtube_video.py
      web.py
      audio.py
      text_content.py

  frontend/
    src/app/
      components/
      contexts/
      routes.tsx

  services/
    research_service/
    roadmap_service/

  data/
    benchmarks/
    processed/

  docs/
    research_pipeline_ai_news_paper.md

  pyproject.toml
  uv.lock
  README.md
```

## Environment Variables

Create a `.env` file in the project root. The project includes `.env.example` with the expected keys.

```env
ASSEMBLYAI_API_KEY=
GOOGLE_API_KEY=
GROQ_API_KEY=
FIRECRAWL_API_KEY=
SUPABASE_JWT_SECRET=
SUPABASE_URL=
SUPABASE_ANON_KEY=
ZEP_API_KEY=
TAVILY_API_KEY=
GITHUB_TOKEN=
NEWS_API_KEY=
```

Notes:

- `SUPABASE_URL` and `SUPABASE_ANON_KEY` are required for authentication and database access.
- `ZEP_API_KEY` is required for notebook memory.
- `GROQ_API_KEY` is used by roadmap generation, query expansion, recursive analysis, graph merging, benchmarking, and final answer synthesis.
- `GOOGLE_API_KEY` is used for Gemini-based summarization/audit flows.
- `TAVILY_API_KEY` is used for search baseline evaluation and optional live evidence reinforcement.
- `GITHUB_TOKEN` and `NEWS_API_KEY` improve AI News ingestion.
- Redis should be running locally on `localhost:6379` for AI News scheduling, deduplication, and cached feed fallback.

## Installation

### Backend

This project uses `uv` with Python `>=3.13`.

```bash
uv sync
```

If you prefer pip-style setup, create a virtual environment and install from `pyproject.toml` using your preferred tooling.

Start the FastAPI server:

```bash
python -m uvicorn backend.app:app --reload
```

The backend runs by default at:

```text
http://localhost:8000
```

Health/docs:

```text
http://localhost:8000/docs
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

The frontend usually runs at:

```text
http://localhost:5173
```

Build the frontend:

```bash
npm run build
```

## Running Redis

The AI News pipeline expects Redis at:

```text
localhost:6379
```

Redis is used for:

- live AI feed sliding window
- source deduplication
- rate/activity tracking
- query-active pause control
- cached AI News fallback when in-memory state is empty

## API Overview

All protected routes expect:

```http
Authorization: Bearer <supabase_access_token>
```

Notebook routes may also use:

```http
X-Notebook-Id: <notebook_id>
```

### Notebook and Sources

```text
POST /api/documents
POST /api/youtube
POST /api/web
POST /api/audio
POST /api/text
GET  /api/notebooks
POST /api/notebooks/{notebook_id}/chat
```

Exact payloads depend on the source type.

### Pathfinder AI

```text
POST  /api/roadmap/generate
GET   /api/roadmap/user
GET   /api/roadmap/{roadmap_id}
GET   /api/roadmap/{roadmap_id}/node/{node_id}/content
PATCH /api/roadmap/{roadmap_id}/node/{node_id}/status?status=done
```

Generate roadmap example:

```json
{
  "topic": "Agentic AI systems",
  "level": "beginner"
}
```

### AI News

```text
GET  /ai-news/get
POST /ai-news/query
```

Query example:

```json
{
  "query": "difference between generative ai and agentic ai",
  "mode": "deep_research"
}
```

The AI News query endpoint automatically writes benchmark results to:

```text
data/benchmarks/query_evaluations.json
data/benchmarks/latest_query_evaluation.json
```

## Benchmarking and Evaluation

FusionAI automatically evaluates AI News query responses against Tavily.

The judge receives:

- question
- Tavily answer
- Tavily supporting text
- FusionAI answer
- FusionAI evidence text

Metrics:

1. Correctness
2. Completeness
3. Context coverage
4. Reasoning depth
5. Actionability
6. Hallucination risk
7. Information gain
8. Citation coverage

For all metrics except hallucination risk, higher is better. For hallucination risk, lower is better.

The benchmark JSON is text-only and avoids dumping embeddings or full graph internals.

## Dataset and Fine-Tuning Support

The research pipeline writes processed examples under:

```text
data/processed/
```

The project includes tooling and dependencies for:

- supervised dataset generation
- Qwen-style fine-tuning experimentation
- PEFT/LoRA adapters
- quantized local model loading with `ctransformers`

These pieces are foundations for reducing dependence on external API providers over time.

## Important Implementation Notes

- The backend uses Supabase authentication middleware for most routes.
- AI News feed data is held in memory and backed by Redis sliding-window fallback.
- Pathfinder node content is generated lazily when a node is opened.
- AI News final answers are generated from retrieved evidence and optional live web reinforcement.
- Frontend benchmark/evaluation internals are intentionally hidden from the user.
- Some generated datasets and benchmark JSON files are local artifacts and may change after running queries.

## Troubleshooting

### Backend returns `401 Missing token`

Make sure the frontend is logged in through Supabase and sends:

```http
Authorization: Bearer <access_token>
```

### AI News feed is empty

Check:

- Redis is running on `localhost:6379`
- `GITHUB_TOKEN` and `NEWS_API_KEY` are configured
- background scheduler has run at least once
- `/ai-news/get` can read from Redis fallback

### Roadmap node opens but content looks empty

The backend and frontend normalize generated content, but if generation fails completely, verify:

- `GROQ_API_KEY` is configured
- the node is not locked
- `/api/roadmap/{roadmap_id}/node/{node_id}/content` returns `content`

### YouTube processing fails

Check:

- the video has accessible transcript/captions
- `yt-dlp` and transcript dependencies are installed
- the URL is public and not region/age restricted

### Supabase access token

The frontend obtains the access token from:

```ts
supabase.auth.getSession()
```

The backend validates that token against Supabase Auth.

## Development Commands

Backend:

```bash
python -m uvicorn backend.app:app --reload
```

Frontend:

```bash
cd frontend
npm run dev
```

Frontend build:

```bash
cd frontend
npm run build
```

Python syntax check:

```bash
python -m py_compile backend/app.py
```

## Resume-Ready Summary

- Built FusionAI, a full-stack AI learning and research platform combining an AI notebook, Pathfinder AI roadmap generation, and a live AI intelligence feed.
- Designed an advanced Graph-RAG research pipeline with semantic chunking, Supabase graph storage, hybrid retrieval, relevance gating, recursive analysis, and evidence-grounded answer synthesis.
- Implemented authenticated multi-source notebook ingestion for documents, YouTube, web pages, audio, and text using Supabase, Zep Cloud, vector retrieval, and LLM-based querying.
- Developed Pathfinder AI for dependency-aware roadmap generation, node progress tracking, lazy educational content generation, and interactive roadmap visualization.
- Added automatic FusionAI-vs-Tavily benchmarking with Groq LLM judging across correctness, completeness, reasoning depth, actionability, information gain, citation coverage, and hallucination risk.

## License

This repository currently includes a `LICENSE` file. Review it before distributing or deploying the project.
