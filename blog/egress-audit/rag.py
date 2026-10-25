"""Local RAG application for the egress audit.

Uses Ollama for inference, Hugging Face for embeddings,
VectorAI DB for retrieval, and Phoenix for tracing.

The application explicitly configures local models and tracing.
Offline restrictions are applied through the hardened
Docker Compose configuration.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

from llama_index.core import (
    Settings,
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.actian_vectorai import ActianVectorAIVectorStore
from openinference.instrumentation.llama_index import LlamaIndexInstrumentor
from phoenix.otel import register

# Service endpoints default to localhost. Inside the Docker Compose network,
# the compose file overrides them with the service names.
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
VECTORAI_URL = os.environ.get("VECTORAI_URL", "localhost:6574")
PHOENIX_TRACES_URL = os.environ.get(
    "PHOENIX_TRACES_URL", "http://localhost:6006/v1/traces"
)

LLM_MODEL = "llama3.2:3b"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
COLLECTION_NAME = "egress_audit"
TOP_K = 3
PHOENIX_PROJECT = "egress-audit"

logger = logging.getLogger("rag")


def configure_tracing() -> None:
    """Send OpenTelemetry traces for every LlamaIndex step to the local Phoenix server."""
    tracer_provider = register(
        project_name=PHOENIX_PROJECT,
        endpoint=PHOENIX_TRACES_URL,
    )
    LlamaIndexInstrumentor().instrument(tracer_provider=tracer_provider)


def configure_models() -> None:
    """Register the local LLM and embedding model as LlamaIndex defaults."""
    Settings.llm = Ollama(
        model=LLM_MODEL,
        base_url=OLLAMA_URL,
        request_timeout=300.0,
        context_window=4096,
    )
    Settings.embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL)


def answer(docs_dir: Path, question: str) -> str:
    """Ingest every file in docs_dir into VectorAI DB and answer one question."""
    documents = SimpleDirectoryReader(input_dir=str(docs_dir)).load_data()
    logger.info("Loaded %d documents from %s", len(documents), docs_dir)

    with ActianVectorAIVectorStore(
        url=VECTORAI_URL,
        collection_name=COLLECTION_NAME,
        stores_text=True,
        clear_existing_collection=True,
    ) as vector_store:
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex.from_documents(
            documents,
            storage_context=storage_context,
        )
        logger.info("Indexed documents into collection %r", COLLECTION_NAME)

        response = index.as_query_engine(similarity_top_k=TOP_K).query(question)

    return str(response)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--docs",
        type=Path,
        default=Path("docs"),
        help="Directory of documents to ingest (default: ./docs).",
    )
    parser.add_argument(
        "--question",
        required=True,
        help="Question to ask against the ingested documents.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = parse_args(argv)

    if not args.docs.is_dir():
        logger.error("Documents directory not found: %s", args.docs)
        return 1

    configure_tracing()
    configure_models()
    print(answer(args.docs, args.question))
    return 0


if __name__ == "__main__":
    sys.exit(main())
