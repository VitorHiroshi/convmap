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

    def embed_text(self, text: str) -> np.ndarray:
        """Embed a raw text string. Useful for concept queries."""
        return self.model.encode(text, normalize_embeddings=True)

    def embed(self, conversation: Conversation) -> EmbeddedConversation:
        """Embed a conversation into adaptive chunks."""
        text = self._build_text(conversation)
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
            text = self._build_text(conv)
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

    def _build_text(self, conversation: Conversation) -> str:
        """Build embeddable text from turns + metadata.

        Metadata fields like category, summary, and output are folded
        into the text so they contribute to the embedding without
        requiring a fixed schema. Timestamps and numeric-only fields
        are skipped (they don't embed well).
        """
        parts = [self._flatten_turns(conversation.turns)]

        meta_text = self._flatten_metadata(conversation.metadata)
        if meta_text:
            parts.append(meta_text)

        return "\n".join(parts)

    def _flatten_turns(self, turns: list[Turn]) -> str:
        """Concatenate turns with speaker labels preserved."""
        return "\n".join(f"{t.speaker}: {t.text}" for t in turns)

    @staticmethod
    def _flatten_metadata(metadata: dict) -> str:
        """Extract embeddable text from metadata.

        Includes string values that carry semantic meaning.
        Skips timestamps, pure numbers, and IDs.
        """
        if not metadata:
            return ""

        skip_keys = {"id", "timestamp", "created_at", "updated_at", "duration", "call_id"}
        parts = []

        for key, value in metadata.items():
            if key.lower() in skip_keys:
                continue
            if not isinstance(value, str):
                continue
            value = value.strip()
            if not value:
                continue
            # Skip values that are just numbers or IDs
            if value.replace(".", "").replace("-", "").isdigit():
                continue
            if len(value) < 3:
                continue
            parts.append(f"{key}: {value}")

        return "\n".join(parts)

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
