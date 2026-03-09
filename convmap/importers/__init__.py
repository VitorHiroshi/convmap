"""Importers convert external transcript formats into Conversations.

Each importer module exposes a `load` function that takes a file path
or raw data and returns a list of Conversations ready for the embedder.
"""
