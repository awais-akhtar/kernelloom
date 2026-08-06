# Retrieval-augmented generation (RAG)

KernelLoom includes a complete local RAG pipeline: load documents, split them,
create embeddings, store vectors, retrieve relevant chunks, build a guarded
context prompt, and generate a cited answer. The core pipeline has no mandatory
database or orchestration dependency.

## Install and choose models

RAG normally uses two models:

- an embedding model converts documents and questions to vectors;
- a chat model writes the final answer from retrieved context.

For local GGUF models, install the llama.cpp extra:

```bash
pip install "kernelloom[llama]"
```

For a CPU-optimized ONNX embedding model plus native FAISS retrieval, install
the RAG extra. Both dependencies stay local and are optional:

```bash
pip install "kernelloom[rag]"
```

## Five-minute persistent RAG

```python
from kernelloom import KernelLoomModel, ModelConfig, RAGConfig, RAGPipeline

chat = KernelLoomModel(ModelConfig(
    "./models/qwen2.5-3b-instruct.gguf",
    model_id="chat",
    context_length=8192,
    temperature=0.1,
))
embeddings = KernelLoomModel(ModelConfig(
    "./models/nomic-embed-text-v1.5.gguf",
    model_id="embeddings",
    embedding=True,
))

rag = RAGPipeline.local(
    chat,
    embeddings,
    database="./data/knowledge.db",  # use "memory" for an ephemeral store
    config=RAGConfig(
        namespace="product-docs",
        chunk_size=900,
        chunk_overlap=120,
        retrieval="mmr",
        top_k=5,
        fetch_k=15,
        min_score=0.15,
        max_context_chars=14_000,
    ),
)

try:
    # A path may be a supported file or a directory. Re-ingestion updates the
    # same deterministic chunk IDs instead of creating duplicates.
    print("Indexed:", rag.ingest("./docs", metadata={"release": "2026.08"}))

    result = rag.ask(
        "How do I configure model batching?",
        filters={"release": "2026.08"},
        generation={"max_new_tokens": 300, "temperature": 0.1},
    )
    print(result.answer)
    for source in result.sources:
        print(source.score, source.document.metadata.get("source"))
finally:
    rag.close()
```

`RAGAnswer.to_dict()` produces a JSON-ready answer, query, source IDs, scores,
and metadata. The generated prompt labels chunks as `[Source 1]`, `[Source 2]`,
and so on so a capable chat model can cite them.

## Input varieties

The built-in loader accepts:

- inline text;
- `Document` objects;
- `.txt`, `.md`, and `.rst` files;
- `.json`, `.jsonl`, and `.csv` files;
- directories, scanned recursively for the supported formats;
- any iterable containing a mixture of those inputs.

```python
from kernelloom import Document

rag.ingest("A short piece of inline knowledge")
rag.ingest("./handbook.md", metadata={"department": "support"})
rag.ingest(["./policies", "./catalog.csv"])
rag.ingest(Document(
    "Refunds are accepted for 30 days.",
    id="refund-policy-v3",
    metadata={"department": "billing", "version": 3},
))

# Isolate tenants or collections in one database.
rag.ingest("./tenant-a", namespace="tenant-a")
answer = rag.ask("What is the refund window?", namespace="tenant-a")
```

For PDF, HTML, OCR, object storage, or application records, convert content to
`Document` objects or provide a custom loader with a compatible `load()` method.

## Retrieval modes

`retrieval="similarity"` returns the highest cosine-similarity scores.
`retrieval="mmr"` first fetches `fetch_k` candidates and applies maximal
marginal relevance to balance query relevance with diversity. `mmr_lambda=1`
favors relevance; lower values favor less repetitive context.

```python
results = rag.retrieve(
    "database backup policy",
    filters={"department": "platform"},
    top_k=8,
)
for item in results:
    print(item.score, item.document.text[:100])
```

Metadata filters use exact equality in the built-in stores. A custom vector
store can interpret filters using its database's native filter language.

## Fast CPU embeddings and FAISS search

The standard-library SQLite store is portable and persistent. For bigger local
collections, use `FastEmbedEmbedder` (local ONNX Runtime models) and
`FaissVectorStore` (native CPU similarity search):

```python
from kernelloom import FaissVectorStore, FastEmbedEmbedder, RAGConfig, RAGPipeline

embeddings = FastEmbedEmbedder(
    "BAAI/bge-small-en-v1.5",
    threads=4,
    batch_size=128,
)
rag = RAGPipeline(
    chat_model,
    embeddings,
    store=FaissVectorStore(metric="cosine"),
    config=RAGConfig(namespace="manuals", retrieval="similarity", top_k=6),
)
rag.ingest("./manuals", batch_size=128)
rag.warmup(["How do I install this product?"])
print(rag.ask("How do I install this product?").answer)
```

`RAGPipeline.local(..., database="faiss")` is a convenient alternative when
using a KernelLoom GGUF embedding model. FAISS indexes are in-memory by design;
persist source documents and rebuild at startup, or implement `VectorStore` for
a persistent database. Use FAISS without metadata filters for the fastest path;
when filters are supplied the adapter scans its namespace's candidates to keep
filter results correct.

FastEmbed can download a selected model on first use. For strictly offline or
air-gapped use, pre-populate its local model cache or use a GGUF embedding model
already present on disk.

## Warm RAG and repeat-query cache

Models remain resident until closed. `RAGPipeline.warmup()` can prime the
generator, embedding backend, FAISS index, and known startup queries before
serving traffic:

```python
startup = rag.warmup([
    "What does this documentation cover?",
    "How do I contact support?",
])
print(startup)
print(rag.cache_info())
```

`RAGConfig.query_cache_size` and `query_cache_ttl_seconds` control a bounded
exact-query retrieval cache (defaults: 256 entries and 30 seconds). Ingestion
and clearing a namespace invalidate it, so freshly indexed content is visible
immediately. KernelLoom model embedding/token caches use separate compact,
bounded LRU storage; see the [CPU-first guide](CPU_FIRST.md) for sizing.

## Use a custom embedding provider

The pipeline recognizes KernelLoom's `embed`/`embed_many` interface and
LangChain's `embed_query`/`embed_documents` interface. Any compatible object
works:

```python
class MyEmbeddings:
    def embed_query(self, text: str) -> list[float]:
        return my_embedding_api([text])[0]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return my_embedding_api(texts)

rag = RAGPipeline(chat_model, MyEmbeddings())
```

Keep the same embedding model and dimensions for indexing and querying. If the
embedding model changes, clear and rebuild that namespace.

## Plug in a custom vector database

Implement four small operations to connect pgvector, Qdrant, Pinecone, Milvus,
Weaviate, Elasticsearch, or an application-specific database:

```python
from collections.abc import Mapping, Sequence
from kernelloom import Document, RAGPipeline, SearchResult

class MyVectorDatabase:
    def upsert(
        self,
        records: Sequence[tuple[Document, Sequence[float]]],
        *,
        namespace: str,
    ) -> int:
        # Write text, metadata, IDs, and vectors in a transaction.
        ...

    def search(
        self,
        embedding: Sequence[float],
        *,
        limit: int,
        namespace: str,
        filters: Mapping[str, object] | None = None,
    ) -> list[SearchResult]:
        # Return best-first results. Include each stored embedding when using MMR.
        ...

    def delete(self, *, namespace: str, ids: Sequence[str] | None = None) -> int:
        ...

    def count(self, *, namespace: str) -> int:
        ...

store = MyVectorDatabase()
rag = RAGPipeline(chat_model, embedding_provider, store=store)
```

For `retrieval="mmr"`, each returned `SearchResult` must include its embedding.
Stores that do not return vectors should use `retrieval="similarity"`.

## Customize splitting and loading

Supply objects with `split(Document) -> list[Document]` and
`load(source, metadata=...) -> list[Document]`:

```python
rag = RAGPipeline(
    chat_model,
    embedding_provider,
    store=custom_store,
    loader=my_pdf_loader,
    splitter=my_token_aware_splitter,
    config=RAGConfig(prompt_template=(
        "{system}\n\nVerified evidence:\n{context}\n\n"
        "User question: {question}\nCited response:"
    )),
)
```

The prompt template must include `{system}`, `{context}`, and `{question}`.
Keep `max_context_chars` comfortably below the chat model's context window so
the question and generated answer still fit.

## Async services

Async methods move synchronous loaders, databases, and local model work off the
event-loop thread. If the generator exposes `ainvoke`, `aask` uses it directly.

```python
import asyncio

async def main():
    indexed = await rag.aingest("./knowledge")
    matches = await rag.aretrieve("How are backups retained?")
    answer = await rag.aask("How are backups retained?")
    print(indexed, matches, answer.answer)

asyncio.run(main())
```

## Store lifecycle

```python
print(rag.count())
deleted = rag.clear()                     # current configured namespace
deleted = rag.clear(namespace="tenant-a")
rag.close()
```

`close()` closes the store, embedder, and generator when they expose a `close`
method. `KernelLoomEmbedder` does not close a supplied embedding model unless it
was created with `close_model=True`; `RAGPipeline.local()` therefore leaves that
model's ownership with the application.

## Production checklist

- Use a persistent store and back it up.
- Assign a namespace per tenant, corpus, or embedding-model version.
- Attach access-control metadata during ingestion and enforce filters on every query.
- Pin one embedding model/version and rebuild vectors after changing it.
- Evaluate retrieval scores and grounded answers with representative questions.
- Tune chunking for the document type instead of assuming one size fits all.
- Set `min_score` so unrelated context is rejected.
- Keep the default instruction that retrieved context is data, not instructions.
- Treat citations as traceability aids; verify critical answers against the source.
