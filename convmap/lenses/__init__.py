"""Lenses are read-only query functions over the map state.

Each lens module exposes functions that take a MapState and return results.
Adding a new lens is just a new module — zero changes to the pipeline.

Usage:
    from convmap.lenses import density

    state = engine.state
    clusters = density.clusters(state)
    emerging = density.emerging(state)
    nearest = density.nearest(state, some_vector, k=5)
"""
