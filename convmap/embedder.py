from __future__ import annotations

import numpy as np

from .types import Turn, Chunk, Conversation, EmbeddedConversation


class Embedder:
    """Converts conversations to chunk embeddings using adaptive windowing.

    Short conversations (< window_size tokens) become a single chunk.
    Longer ones get a sliding window with overlap, so every sentence
    appears complete in at least one chunk regardless of turn boundaries.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        window_size: int = 200,
        overlap: float = 0.5,
    ):
        # Lazy import — only pay the cost when actually embedding
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)
        self.window_size = window_size
        self.overlap = overlap

    def embed(self, conversation: Conversation) -> EmbeddedConversation:
        """Embed a conversation into adaptive chunks."""
        text = self._flatten_turns(conversation.turns)
        chunks = self._adaptive_chunk(text)

        texts = [c.text for c in chunks]
        embeddings = self.model.encode(texts, normalize_embeddings=True)

        for chunk, emb in zip(chunks, embeddings):
            chunk.embedding = emb

        return EmbeddedConversation(
            id=conversation.id,
            chunks=chunks,
            metadata=conversation.metadata,
        )

    def embed_batch(self, conversations: list[Conversation]) -> list[EmbeddedConversation]:
        """Embed multiple conversations, batching all chunks through the model at once."""
        all_chunks: list[list[Chunk]] = []
        all_texts: list[str] = []

        for conv in conversations:
            text = self._flatten_turns(conv.turns)
            chunks = self._adaptive_chunk(text)
            all_chunks.append(chunks)
            all_texts.extend(c.text for c in chunks)

        if not all_texts:
            return []

        all_embeddings = self.model.encode(all_texts, normalize_embeddings=True)

        results = []
        offset = 0
        for conv, chunks in zip(conversations, all_chunks):
            for chunk in chunks:
                chunk.embedding = all_embeddings[offset]
                offset += 1
            results.append(EmbeddedConversation(
                id=conv.id,
                chunks=chunks,
                metadata=conv.metadata,
            ))

        return results

    def _flatten_turns(self, turns: list[Turn]) -> str:
        """Concatenate turns with speaker labels preserved."""
        return "\n".join(f"{t.speaker}: {t.text}" for t in turns)

    def _adaptive_chunk(self, text: str) -> list[Chunk]:
        """Split text into chunks, adapting to conversation length."""
        tokens = text.split()

        if len(tokens) <= self.window_size:
            return [Chunk(text=text, index=0)]

        chunks = []
        step = max(1, int(self.window_size * (1 - self.overlap)))

        for i, start in enumerate(range(0, len(tokens), step)):
            end = start + self.window_size
            chunk_tokens = tokens[start:end]
            if not chunk_tokens:
                break
            chunks.append(Chunk(text=" ".join(chunk_tokens), index=i))
            if end >= len(tokens):
                break

        return chunks
