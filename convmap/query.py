"""Query interface — thin coordination layer over lenses.

Adds time filtering as a cross-cutting concern. All math
lives in the lenses; queries compose them.
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np

from .types import MapState
from .lenses import density, neighborhood


def concept(
    state: MapState,
    query_embedding: np.ndarray,
    k: int = 10,
    time_range: tuple[float, float] | None = None,
) -> list[dict]:
    """Find recent vectors most similar to a concept embedding.

    Delegates to neighborhood.similar with optional time filtering.
    """
    scoped = time_window(state, *time_range) if time_range else state
    return neighborhood.similar(scoped, query_embedding, k)


def time_window(
    state: MapState,
    start: float,
    end: float,
) -> MapState:
    """Return a filtered MapState containing only vectors within a time range.

    The returned state can be passed to any lens for time-scoped analysis.
    """
    filtered = _filter_by_time(state.recent_vectors, (start, end))
    filtered_outliers = _filter_by_time(state.outlier_buffer, (start, end))

    return MapState(
        core_clusters=state.core_clusters,
        potential_clusters=state.potential_clusters,
        outlier_buffer=filtered_outliers,
        recent_vectors=filtered,
        snapshots=state.snapshots,
        dimensions=state.dimensions,
        timestamp=state.timestamp,
    )


def funnel(state: MapState) -> list[dict]:
    """Cluster funnel — delegates to density.funnel."""
    return density.funnel(state)


def compare_windows(
    state: MapState,
    window_a: tuple[float, float],
    window_b: tuple[float, float],
) -> dict:
    """Compare two time windows by cluster distribution shift.

    Uses time_window for filtering, density.cluster_distribution for math.
    """
    state_a = time_window(state, *window_a)
    state_b = time_window(state, *window_b)

    vecs_a = state_a.recent_vectors
    vecs_b = state_b.recent_vectors

    if not vecs_a or not vecs_b or not state.core_clusters:
        return {
            "window_a_count": len(vecs_a),
            "window_b_count": len(vecs_b),
            "cluster_shifts": [],
        }

    dist_a = density.cluster_distribution(state, vecs_a)
    dist_b = density.cluster_distribution(state, vecs_b)

    shifts = []
    for i in range(len(state.core_clusters)):
        share_a = dist_a[i]
        share_b = dist_b[i]
        delta = share_b - share_a
        shifts.append({
            "cluster_index": i,
            "label": state.core_clusters[i].label,
            "share_a": share_a,
            "share_b": share_b,
            "delta": delta,
        })

    shifts.sort(key=lambda x: abs(x["delta"]), reverse=True)

    return {
        "window_a_count": len(vecs_a),
        "window_b_count": len(vecs_b),
        "cluster_shifts": shifts,
    }


def segment(
    state: MapState,
    cluster_index: int,
    time_range: tuple[float, float] | None = None,
) -> list[dict]:
    """Get members of a cluster — delegates to density.segment with time filter."""
    scoped = time_window(state, *time_range) if time_range else state
    return density.segment(scoped, cluster_index)


def histogram(
    state: MapState,
    time_range: tuple[float, float] | None = None,
) -> dict:
    """Vector distribution across clusters — delegates to density.histogram."""
    scoped = time_window(state, *time_range) if time_range else state
    return density.histogram(scoped)


def anomalies(
    state: MapState,
    threshold: float = 0.5,
    time_range: tuple[float, float] | None = None,
) -> list[dict]:
    """Find vectors that don't fit any cluster — delegates to density.anomalies."""
    scoped = time_window(state, *time_range) if time_range else state
    return density.anomalies(scoped, threshold)


def report(state: MapState, anomaly_threshold: float = 0.5) -> dict:
    """Build a full map report combining all lenses.

    Returns a dict with sections: summary, clusters, funnel, histogram,
    emerging, anomalies, drift, topology, and cluster_details.

    Designed as a single-call entry point for LLM consumers.
    """
    from collections import Counter
    from .lenses import drift as drift_lens
    from .lenses import topology

    dist = density.distribution(state)
    cls = density.clusters(state)
    hist = density.histogram(state)
    funnel_data = density.funnel(state)
    emerging_data = density.emerging(state)
    anomaly_data = density.anomalies(state, anomaly_threshold)

    drift_data = []
    if len(state.snapshots) >= 2:
        drift_data = drift_lens.detect(state.snapshots)

    # Topology
    edges = topology.adjacency(state)
    iso = topology.isolated(state)
    dmap = topology.density_map(state)
    bridge_count = len(topology.bridges(state))

    # Per-cluster details with metadata aggregation
    cluster_details = []
    for i in range(len(state.core_clusters)):
        members = density.segment(state, i)
        meta_stats = _aggregate_metadata(members)
        samples = [
            {"similarity": m["similarity"], "metadata": m["metadata"]}
            for m in members[:5]
        ]
        cluster_details.append({
            "index": i,
            "label": state.core_clusters[i].label,
            "member_count": len(members),
            "metadata_stats": meta_stats,
            "top_members": samples,
        })

    return {
        "summary": {
            "dimensions": state.dimensions,
            "core_clusters": dist["core_count"],
            "potential_clusters": dist["potential_count"],
            "outliers": dist["outlier_count"],
            "recent_vectors": len(state.recent_vectors),
            "snapshots": len(state.snapshots),
            "core_total_weight": dist["core_total_weight"],
        },
        "clusters": [
            {
                "label": c["label"],
                "weight": c["weight"],
                "count": c["count"],
                "radius": c["radius"],
            }
            for c in cls
        ],
        "funnel": [
            {
                "rank": s["rank"],
                "label": s["label"],
                "count": s["count"],
                "share": s["share"],
                "cumulative_share": s["cumulative_share"],
            }
            for s in funnel_data
        ],
        "histogram": {
            "total": hist["total"],
            "unassigned": hist["unassigned"],
            "clusters": [
                {
                    "index": c["index"],
                    "label": c["label"],
                    "count": c["count"],
                    "share": c["share"],
                }
                for c in hist.get("clusters", [])
            ],
        },
        "emerging": [
            {
                "weight": e["weight"],
                "count": e["count"],
                "momentum": e["momentum"],
            }
            for e in emerging_data[:10]
        ],
        "anomalies": {
            "count": len(anomaly_data),
            "threshold": anomaly_threshold,
            "items": [
                {
                    "id": a["metadata"].get("id", "?"),
                    "best_similarity": a["best_similarity"],
                    "best_cluster": a["best_cluster"],
                }
                for a in anomaly_data[:20]
            ],
        },
        "drift": [
            {
                "from_label": e["from_label"],
                "to_label": e["to_label"],
                "magnitude": e["magnitude"],
                "appeared": e["diff"]["appeared"],
                "disappeared": e["diff"]["disappeared"],
                "moved": len(e["diff"]["moved"]),
                "grew": len(e["diff"]["grew"]),
                "shrunk": len(e["diff"]["shrunk"]),
            }
            for e in drift_data
        ],
        "topology": {
            "adjacency": [
                {"i": e["i"], "j": e["j"], "similarity": e["similarity"],
                 "label_i": e["label_i"], "label_j": e["label_j"]}
                for e in edges
            ],
            "bridges_count": bridge_count,
            "isolated": [
                {"index": c["index"], "label": c["label"],
                 "max_similarity": c["max_similarity"]}
                for c in iso
            ],
            "density_map": {
                "mean_density": dmap["mean_density"],
                "std_density": dmap["std_density"],
                "clusters": [
                    {"index": c["index"], "label": c["label"],
                     "density": c["density"], "weight": c["weight"],
                     "radius": c["radius"]}
                    for c in dmap["clusters"]
                ],
            },
        },
        "cluster_details": cluster_details,
    }


def _aggregate_metadata(members: list[dict]) -> dict:
    """Generic metadata aggregation across cluster members."""
    if not members:
        return {}

    from collections import Counter

    key_values: dict[str, list] = {}
    skip_keys = {"id", "n_chunks", "timestamp", "created_at", "updated_at"}

    for m in members:
        for k, v in m["metadata"].items():
            if k.lower() in skip_keys:
                continue
            key_values.setdefault(k, []).append(v)

    stats = {}
    for key, values in key_values.items():
        numeric_vals = []
        for v in values:
            try:
                numeric_vals.append(float(v))
            except (ValueError, TypeError):
                pass

        if numeric_vals and len(numeric_vals) > len(values) * 0.5:
            sorted_nums = sorted(numeric_vals)
            n = len(sorted_nums)
            stats[key] = {
                "type": "numeric",
                "count": n,
                "min": round(sorted_nums[0], 2),
                "max": round(sorted_nums[-1], 2),
                "mean": round(sum(sorted_nums) / n, 2),
                "median": round(sorted_nums[n // 2], 2),
            }
        else:
            counter = Counter(str(v) for v in values)
            total = len(values)
            stats[key] = {
                "type": "categorical",
                "count": total,
                "unique": len(counter),
                "top": [
                    {"value": v, "count": c, "share": round(c / total, 4)}
                    for v, c in counter.most_common(10)
                ],
            }

    return stats


def report_markdown(data: dict) -> str:
    """Render a report dict as Markdown."""
    lines = []
    s = data["summary"]

    lines.append("# Convmap Report")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"| Metric | Value |")
    lines.append(f"|---|---|")
    lines.append(f"| Dimensions | {s['dimensions']} |")
    lines.append(f"| Core clusters | {s['core_clusters']} |")
    lines.append(f"| Potential clusters | {s['potential_clusters']} |")
    lines.append(f"| Outliers | {s['outliers']} |")
    lines.append(f"| Recent vectors | {s['recent_vectors']} |")
    lines.append(f"| Snapshots | {s['snapshots']} |")
    lines.append(f"| Core total weight | {s['core_total_weight']:.1f} |")
    lines.append("")

    if data["clusters"]:
        lines.append("## Clusters")
        lines.append("")
        lines.append("| # | Label | Weight | Count | Radius |")
        lines.append("|---|---|---|---|---|")
        for i, c in enumerate(data["clusters"]):
            label = c["label"] or f"cluster-{i}"
            lines.append(f"| {i} | {label} | {c['weight']:.1f} | {c['count']} | {c['radius']:.4f} |")
        lines.append("")

    if data["funnel"]:
        lines.append("## Funnel")
        lines.append("")
        lines.append("| Rank | Label | Count | Share | Cumulative |")
        lines.append("|---|---|---|---|---|")
        for f in data["funnel"]:
            label = f["label"] or f"cluster-{f['rank']}"
            lines.append(
                f"| {f['rank']} | {label} | {f['count']} "
                f"| {f['share']:.1%} | {f['cumulative_share']:.1%} |"
            )
        lines.append("")

    hist = data["histogram"]
    if hist["total"] > 0:
        lines.append("## Histogram")
        lines.append("")
        lines.append(f"Total vectors: {hist['total']}, unassigned: {hist['unassigned']}")
        lines.append("")
        lines.append("| Index | Label | Count | Share |")
        lines.append("|---|---|---|---|")
        for c in hist["clusters"]:
            label = c["label"] or f"cluster-{c['index']}"
            lines.append(f"| {c['index']} | {label} | {c['count']} | {c['share']:.1%} |")
        lines.append("")

    if data["emerging"]:
        lines.append("## Emerging Patterns")
        lines.append("")
        lines.append("| Weight | Count | Momentum |")
        lines.append("|---|---|---|")
        for e in data["emerging"]:
            lines.append(f"| {e['weight']:.1f} | {e['count']} | {e['momentum']:.4f} |")
        lines.append("")

    anom = data["anomalies"]
    if anom["count"] > 0:
        lines.append(f"## Anomalies (threshold={anom['threshold']})")
        lines.append("")
        lines.append(f"Found: {anom['count']}")
        lines.append("")
        lines.append("| # | ID | Best Similarity | Nearest Cluster |")
        lines.append("|---|---|---|---|")
        for i, a in enumerate(anom["items"]):
            lines.append(f"| {i + 1} | {a['id']} | {a['best_similarity']:.4f} | {a['best_cluster']} |")
        if anom["count"] > 20:
            lines.append(f"| | ... {anom['count'] - 20} more | | |")
        lines.append("")

    if data["drift"]:
        lines.append("## Drift")
        lines.append("")
        lines.append("| From | To | Magnitude | Appeared | Disappeared | Moved | Grew | Shrunk |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for d in data["drift"]:
            lines.append(
                f"| {d['from_label'] or '?'} | {d['to_label'] or '?'} "
                f"| {d['magnitude']} | {d['appeared']} | {d['disappeared']} "
                f"| {d['moved']} | {d['grew']} | {d['shrunk']} |"
            )
        lines.append("")

    return "\n".join(lines)


# --- Cross-cutting helper ---


def _filter_by_time(
    vectors: list[tuple[np.ndarray, dict]],
    time_range: tuple[float, float] | None,
) -> list[tuple[np.ndarray, dict]]:
    """Filter vectors by timestamp in metadata.

    Looks for 'timestamp' or 'created_at' keys.
    Handles both numeric (epoch) and ISO 8601 string timestamps.
    """
    if time_range is None:
        return vectors

    start, end = time_range
    filtered = []
    for vec, meta in vectors:
        t = _extract_timestamp(meta)
        if t is not None and start <= t <= end:
            filtered.append((vec, meta))
    return filtered


def _extract_timestamp(meta: dict) -> float | None:
    """Extract a numeric timestamp from metadata.

    Checks 'timestamp' then 'created_at'. Parses epoch floats and ISO strings.
    """
    for key in ("timestamp", "created_at"):
        raw = meta.get(key)
        if raw is None:
            continue
        try:
            return float(raw)
        except (ValueError, TypeError):
            pass
        if isinstance(raw, str):
            try:
                dt = datetime.fromisoformat(_normalize_tz(raw))
                return dt.timestamp()
            except (ValueError, TypeError):
                pass
    return None


def _normalize_tz(s: str) -> str:
    """Normalize timezone offset for Python 3.9 fromisoformat compatibility.

    Converts '+00' to '+00:00', '-03' to '-03:00', etc.
    """
    import re
    return re.sub(r'([+-]\d{2})$', r'\1:00', s)
