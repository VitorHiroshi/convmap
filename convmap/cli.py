"""CLI for convmap — import data, query the map, manage persistence.

Usage:
    convmap import <file> [options]
    convmap query <lens> [args]
    convmap status
    convmap snapshot [--label <label>]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

DEFAULT_MAP = ".convmap"


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(
        prog="convmap",
        description="Streaming conversation map with pluggable lenses",
    )
    parser.add_argument(
        "--map", default=DEFAULT_MAP, help="Path to map directory (default: .convmap)"
    )
    parser.add_argument(
        "--json", action="store_true", dest="json_output",
        help="Output as JSON (for programmatic / LLM consumption)",
    )

    sub = parser.add_subparsers(dest="command")

    # --- import ---
    imp = sub.add_parser("import", help="Import conversations into the map")
    imp.add_argument("file", help="CSV or JSONL file to import")
    imp.add_argument(
        "--format", choices=["csv", "jsonl"], help="File format (auto-detected)"
    )
    imp.add_argument("--id-column", help="Column name for conversation ID")
    imp.add_argument("--transcript-column", help="Column for transcript text")
    imp.add_argument("--speaker-column", help="Column for speaker")
    imp.add_argument("--text-column", help="Column for text in per-turn layout")
    imp.add_argument("--delimiter", default=",", help="CSV delimiter")
    imp.add_argument(
        "--model", default="BAAI/bge-m3", help="Embedding model name"
    )
    imp.add_argument(
        "--batch-size", type=int, default=64, help="Embedding batch size"
    )

    # --- query ---
    qry = sub.add_parser("query", help="Query the map")
    qry.add_argument(
        "--json", action="store_true", dest="json_output",
        help="Output as JSON (for programmatic / LLM consumption)",
    )
    qsub = qry.add_subparsers(dest="lens")

    qc = qsub.add_parser("concept", help="Find conversations similar to a concept")
    qc.add_argument("text", help="Concept text to search for")
    qc.add_argument("--k", type=int, default=10, help="Number of results")
    qc.add_argument("--model", default="BAAI/bge-m3", help="Embedding model")
    qc.add_argument("--time-start", help="Start timestamp (epoch or ISO)")
    qc.add_argument("--time-end", help="End timestamp (epoch or ISO)")

    qsub.add_parser("clusters", help="Show cluster distribution")
    qsub.add_parser("funnel", help="Analyze cluster funnel by size")

    qh = qsub.add_parser("histogram", help="Vector distribution across clusters")
    qh.add_argument("--time-start", help="Start timestamp (epoch or ISO)")
    qh.add_argument("--time-end", help="End timestamp (epoch or ISO)")

    qsub.add_parser("drift", help="Detect drift across snapshots")

    qa = qsub.add_parser("anomalies", help="Find vectors outside all clusters")
    qa.add_argument("--threshold", type=float, default=0.5)
    qa.add_argument("--time-start", help="Start timestamp (epoch or ISO)")
    qa.add_argument("--time-end", help="End timestamp (epoch or ISO)")

    qs = qsub.add_parser("segment", help="Get members of a specific cluster")
    qs.add_argument("index", type=int, help="Cluster index")
    qs.add_argument("--time-start", help="Start timestamp (epoch or ISO)")
    qs.add_argument("--time-end", help="End timestamp (epoch or ISO)")

    qcmp = qsub.add_parser("compare", help="Compare two time windows")
    qcmp.add_argument("start_a", help="Window A start (epoch or ISO)")
    qcmp.add_argument("end_a", help="Window A end (epoch or ISO)")
    qcmp.add_argument("start_b", help="Window B start (epoch or ISO)")
    qcmp.add_argument("end_b", help="Window B end (epoch or ISO)")

    qsub.add_parser("topology", help="Show cluster relationships and bridges")

    qtda = qsub.add_parser("tda", help="Topological features of the map")
    qtda.add_argument(
        "--source", choices=["core", "recent", "all"], default="core",
        help="Point source for TDA (default: core)",
    )

    # --- report ---
    rep = sub.add_parser("report", help="Generate a full map report")
    rep.add_argument(
        "--format", choices=["json", "markdown"], default="json", dest="report_format",
        help="Output format (default: json)",
    )
    rep.add_argument(
        "--output", "-o", help="Write report to file instead of stdout",
    )
    rep.add_argument(
        "--anomaly-threshold", type=float, default=0.5,
        help="Similarity threshold for anomaly detection",
    )

    # --- status ---
    sub.add_parser("status", help="Show map status")

    # --- snapshot ---
    snap = sub.add_parser("snapshot", help="Capture a distribution snapshot")
    snap.add_argument("--label", help="Label for the snapshot")

    # --- label ---
    lbl = sub.add_parser("label", help="Label a core cluster")
    lbl.add_argument("index", type=int, help="Cluster index")
    lbl.add_argument("name", help="Label to assign")

    # --- init ---
    init = sub.add_parser("init", help="Initialize convmap in a project")
    init.add_argument("--data", help="Path to data file (CSV or JSONL) for initial import")
    init.add_argument(
        "--model", default="BAAI/bge-m3", help="Embedding model name"
    )
    init.add_argument("--skip-import", action="store_true", help="Only scaffold, skip initial import")

    # --- top-level lens shortcuts ---
    # Register each lens as a top-level command so "convmap clusters" works
    # --- recluster ---
    qrc = qsub.add_parser("recluster", help="Re-cluster vectors with different parameters")
    qrc.add_argument("--epsilon", type=float, default=0.15, help="Similarity threshold (default: 0.15)")
    qrc.add_argument("--mu", type=float, default=3.0, help="Min weight for core cluster (default: 3)")
    qrc.add_argument("--beta", type=float, default=0.3, help="Survival fraction of mu (default: 0.3)")

    # --- sweep ---
    qsw = qsub.add_parser("sweep", help="Sweep epsilon values to find best granularity")
    qsw.add_argument("--mu", type=float, default=3.0, help="Min weight for core cluster (default: 3)")
    qsw.add_argument(
        "--epsilons", nargs="+", type=float,
        help="Epsilon values to sweep (default: 0.05 0.10 0.15 0.20 0.25 0.30 0.35 0.40)",
    )

    _lens_names = {
        "clusters", "funnel", "histogram", "drift", "anomalies",
        "segment", "compare", "topology", "tda", "concept",
        "recluster", "sweep",
    }
    for _name in _lens_names:
        sub.add_parser(_name, add_help=False)

    args, remaining = parser.parse_known_args(argv)

    if args.command is None:
        parser.print_help()
        return

    # Top-level lens shortcut: "convmap clusters" → "convmap query clusters"
    if args.command in _lens_names:
        query_argv = ["query"]
        # --json may appear in args or in remaining (argparse quirk with subcommands)
        json_flag = args.json_output or "--json" in remaining
        if json_flag:
            query_argv.append("--json")
            remaining = [r for r in remaining if r != "--json"]
        if args.map != DEFAULT_MAP:
            query_argv = ["--map", args.map] + query_argv
        query_argv.append(args.command)
        query_argv.extend(remaining)
        return main(query_argv)

    if args.command == "init":
        _cmd_init(args)
    elif args.command == "import":
        _cmd_import(args)
    elif args.command == "query":
        if args.lens is None:
            qry.print_help()
            return
        _cmd_query(args)
    elif args.command == "report":
        _cmd_report(args)
    elif args.command == "status":
        _cmd_status(args)
    elif args.command == "snapshot":
        _cmd_snapshot(args)
    elif args.command == "label":
        _cmd_label(args)


# ── Init ────────────────────────────────────────────────────────────


def _cmd_init(args):
    import json
    import subprocess

    project_root = _find_project_root(Path.cwd())
    map_dir = Path(args.map)

    print(f"\n── convmap init ──\n")
    print(f"Project root:  {project_root}")
    print(f"Map directory: {map_dir.resolve()}")

    # Scaffold .convmap directory
    map_dir.mkdir(parents=True, exist_ok=True)

    # Detect data files if not specified
    data_file = None
    if args.data:
        data_file = Path(args.data)
        if not data_file.exists():
            print(f"\nData file not found: {data_file}", file=sys.stderr)
            sys.exit(1)
    else:
        candidates = list(project_root.glob("*.csv")) + list(project_root.glob("*.jsonl"))
        if candidates:
            print(f"\nDetected data files:")
            for f in candidates[:10]:
                print(f"  {f.relative_to(project_root)}")
            data_file = candidates[0]
            print(f"\nUsing: {data_file.name}")

    # Add .convmap/ to .gitignore
    gitignore = project_root / ".gitignore"
    if gitignore.exists():
        content = gitignore.read_text()
        if ".convmap/" not in content:
            with gitignore.open("a") as f:
                f.write("\n.convmap/\n")
            print("Added .convmap/ to .gitignore")
    else:
        gitignore.write_text(".convmap/\n")
        print("Created .gitignore with .convmap/")

    # Detect project type and add scripts
    _scaffold_scripts(project_root, data_file, args)

    # Run initial import
    if data_file and not args.skip_import:
        print(f"\nRunning initial import...\n")
        _cmd_import(argparse.Namespace(
            file=str(data_file),
            format=None,
            id_column=None,
            transcript_column=None,
            speaker_column=None,
            text_column=None,
            delimiter=",",
            model=args.model,
            batch_size=64,
            map=args.map,
        ))
    elif not data_file:
        print("\nNo data file found. Import manually:")
        print("  convmap import <file.csv|file.jsonl>")

    # Print next steps
    print(f"\n── Setup complete ──\n")
    print("Commands:")
    print("  convmap status                    # map overview")
    print("  convmap report --format json      # full analysis (one call, all data)")
    print("  convmap query --json clusters     # cluster info")
    print("  convmap query --json segment 0    # cluster members")
    print("  convmap query --json histogram    # distribution")
    print("  convmap snapshot --label v1       # capture snapshot for drift detection")
    print("  convmap label 0 'short-calls'     # name a cluster")
    print()


def _scaffold_scripts(project_root: Path, data_file: Path | None, args):
    """Add convmap scripts to the project's package manager config."""
    import json

    data_flag = f" {data_file.name}" if data_file else " <data-file>"

    # Python project: pyproject.toml
    pyproject = project_root / "pyproject.toml"
    if pyproject.exists():
        content = pyproject.read_text()
        if "[project.scripts]" in content and "convmap" not in content.split("[project.scripts]")[1].split("[")[0]:
            # Has scripts section but no convmap — don't modify, just inform
            pass
        if "convmap" not in content:
            print("\nAdd to your pyproject.toml [project.scripts]:")
            print(f'  convmap = "convmap.cli:main"')
        return

    # Node project: package.json
    pkg_json = project_root / "package.json"
    if pkg_json.exists():
        try:
            pkg = json.loads(pkg_json.read_text())
            scripts = pkg.setdefault("scripts", {})
            changed = False

            if "convmap" not in scripts:
                scripts["convmap"] = f"convmap import{data_flag}"
                changed = True
            if "convmap:report" not in scripts:
                scripts["convmap:report"] = "convmap report --format json"
                changed = True
            if "convmap:query" not in scripts:
                scripts["convmap:query"] = "convmap query --json --"
                changed = True
            if "convmap:status" not in scripts:
                scripts["convmap:status"] = "convmap status"
                changed = True

            if changed:
                pkg_json.write_text(json.dumps(pkg, indent=2) + "\n")
                pm = _detect_package_manager(project_root)
                print(f"\nScripts added to package.json:")
                print(f"  {pm} run convmap          # import data")
                print(f"  {pm} run convmap:report   # full JSON report")
                print(f"  {pm} run convmap:query -- clusters  # query")
                print(f"  {pm} run convmap:status   # map status")
            else:
                print("\nConvmap scripts already in package.json")
        except (json.JSONDecodeError, KeyError):
            pass
        return

    # Makefile fallback
    makefile = project_root / "Makefile"
    if makefile.exists():
        content = makefile.read_text()
        if "convmap" not in content:
            with makefile.open("a") as f:
                f.write(f"\n\n# ── convmap ──\n")
                f.write(f"convmap:\n\tconvmap import{data_flag}\n\n")
                f.write(f"convmap-report:\n\tconvmap report --format json\n\n")
                f.write(f"convmap-query:\n\tconvmap query --json $(ARGS)\n\n")
                f.write(f"convmap-status:\n\tconvmap status\n")
            print("\nTargets added to Makefile: convmap, convmap-report, convmap-query, convmap-status")
        return

    # No project file found — just print instructions
    print("\nNo package.json/pyproject.toml/Makefile found.")
    print("Run commands directly:")
    print(f"  convmap import{data_flag}")
    print(f"  convmap report --format json")


def _find_project_root(start: Path) -> Path:
    """Walk up to find the project root (has package.json, pyproject.toml, or .git)."""
    markers = ["package.json", "pyproject.toml", "setup.py", ".git"]
    current = start
    while current != current.parent:
        if any((current / m).exists() for m in markers):
            return current
        current = current.parent
    return start


def _detect_package_manager(root: Path) -> str:
    if (root / "pnpm-lock.yaml").exists():
        return "pnpm"
    if (root / "bun.lockb").exists() or (root / "bun.lock").exists():
        return "bun"
    if (root / "yarn.lock").exists():
        return "yarn"
    return "npm"


# ── Import ──────────────────────────────────────────────────────────


def _cmd_import(args):
    from . import persistence
    from .embedder import Embedder
    from .engine import Engine

    file_path = Path(args.file)
    if not file_path.exists():
        print(f"File not found: {file_path}", file=sys.stderr)
        sys.exit(1)

    fmt = args.format
    if fmt is None:
        ext = file_path.suffix.lower()
        if ext == ".csv":
            fmt = "csv"
        elif ext in (".jsonl", ".json"):
            fmt = "jsonl"
        else:
            print(
                f"Cannot detect format from '{ext}'. Use --format.",
                file=sys.stderr,
            )
            sys.exit(1)

    print(f"Loading {fmt}: {file_path}")

    if fmt == "csv":
        from .importers.csv_importer import load

        kwargs = {"delimiter": args.delimiter}
        if args.id_column:
            kwargs["id_column"] = args.id_column
        if args.transcript_column:
            kwargs["transcript_column"] = args.transcript_column
        if args.speaker_column:
            kwargs["speaker_column"] = args.speaker_column
        if args.text_column:
            kwargs["text_column"] = args.text_column
        conversations = load(file_path, **kwargs)
    else:
        from .importers.jsonl import load

        conversations = load(file_path)

    print(f"Loaded {len(conversations)} conversations")
    if not conversations:
        return

    map_path = Path(args.map)

    print(f"Embedding with {args.model}")
    embedder = Embedder(model_name=args.model)

    if (map_path / "meta.json").exists():
        print(f"Loading existing map: {map_path}")
        engine = persistence.load(map_path)
    else:
        print(f"Creating new map: {map_path}")
        dims = len(embedder.embed_text("test"))
        engine = Engine(dimensions=dims)

    batch_size = args.batch_size
    total = len(conversations)

    for i in range(0, total, batch_size):
        batch = conversations[i : i + batch_size]
        embedded = embedder.embed_batch(batch)
        for ec in embedded:
            engine.ingest(ec)

        done = min(i + len(batch), total)
        pct = int(done / total * 100)
        print(f"  [{pct:3d}%] {done}/{total}")

    engine._maintain()
    persistence.save(engine, map_path)

    s = engine.summary
    print(f"\nMap saved: {map_path}")
    print(f"  Core clusters:     {s['core_clusters']}")
    print(f"  Potential clusters: {s['potential_clusters']}")
    print(f"  Recent vectors:    {s['recent_vectors']}")
    print(f"  Total ingested:    {s['total_ingested']}")


# ── Query ───────────────────────────────────────────────────────────


def _print_json(data):
    import json

    def _default(obj):
        if hasattr(obj, "tolist"):
            return obj.tolist()
        return str(obj)

    print(json.dumps(data, indent=2, default=_default))


def _cmd_query(args):
    from . import persistence

    map_path = Path(args.map)
    if not (map_path / "meta.json").exists():
        print(f"No map at {map_path}. Run 'convmap import' first.", file=sys.stderr)
        sys.exit(1)

    engine = persistence.load(map_path)
    state = engine.state

    dispatch = {
        "concept": _query_concept,
        "clusters": _query_clusters,
        "funnel": _query_funnel,
        "histogram": _query_histogram,
        "drift": _query_drift,
        "anomalies": _query_anomalies,
        "segment": _query_segment,
        "compare": _query_compare,
        "topology": _query_topology,
        "tda": _query_tda,
        "recluster": _query_recluster,
        "sweep": _query_sweep,
    }

    dispatch[args.lens](args, state)


def _query_concept(args, state):
    from . import query
    from .embedder import Embedder

    embedder = Embedder(model_name=args.model)
    query_vec = embedder.embed_text(args.text)

    time_range = _parse_time_range(args)
    results = query.concept(state, query_vec, k=args.k, time_range=time_range)

    data = {
        "query": args.text,
        "count": len(results),
        "results": [
            {"similarity": r["similarity"], "metadata": r["metadata"]}
            for r in results
        ],
    }

    if args.json_output:
        _print_json(data)
        return

    print(f'\nConcept: "{args.text}"')
    print(f"{len(results)} results:\n")

    for i, r in enumerate(results):
        meta = r["metadata"]
        conv_id = meta.get("id", "?")
        print(f"  {i + 1}. [{r['similarity']:.4f}] {conv_id}")
        for k, v in meta.items():
            if k not in ("id", "n_chunks") and isinstance(v, str) and len(v) > 2:
                print(f"     {k}: {_truncate(v, 100)}")


def _query_clusters(args, state):
    from .lenses import density

    cls = density.clusters(state)
    dist = density.distribution(state)
    emerging_list = density.emerging(state)

    data = {
        "distribution": dist,
        "clusters": [
            {k: v for k, v in c.items() if k != "centroid"} for c in cls
        ],
        "emerging": [
            {k: v for k, v in e.items() if k != "centroid"} for e in emerging_list[:10]
        ],
    }

    if args.json_output:
        _print_json(data)
        return

    print(f"\n{dist['core_count']} core clusters, {dist['potential_count']} potential")
    print(f"Total weight: {dist['core_total_weight']:.1f}\n")

    for i, c in enumerate(cls):
        label = c["label"] or f"cluster-{i}"
        print(f"  {i}. {label}")
        print(f"     weight: {c['weight']:.1f}  count: {c['count']}  radius: {c['radius']:.4f}")

    if emerging_list:
        print(f"\nEmerging ({len(emerging_list)}):")
        for e in emerging_list[:10]:
            print(f"  - weight: {e['weight']:.1f}  count: {e['count']}  momentum: {e['momentum']:.4f}")


def _query_funnel(args, state):
    from . import query

    stages = query.funnel(state)

    data = {
        "stages": [
            {k: v for k, v in s.items() if k != "centroid"} for s in stages
        ],
    }

    if args.json_output:
        _print_json(data)
        return

    if not stages:
        print("No clusters formed yet.")
        return

    print(f"\nFunnel ({len(stages)} stages):\n")
    for s in stages:
        label = s["label"] or f"cluster-{s['rank']}"
        bar = "\u2588" * int(s["share"] * 40)
        print(f"  {s['rank']}. {label}")
        print(f"     count: {s['count']}  share: {s['share']:.1%}  cumul: {s['cumulative_share']:.1%}")
        print(f"     {bar}")


def _query_histogram(args, state):
    from . import query

    time_range = _parse_time_range(args)
    hist = query.histogram(state, time_range=time_range)

    if args.json_output:
        _print_json(hist)
        return

    print(f"\nHistogram: {hist['total']} vectors, {hist['unassigned']} unassigned\n")

    for c in hist["clusters"]:
        label = c["label"] or f"cluster-{c['index']}"
        bar = "\u2588" * int(c["share"] * 40)
        print(f"  {c['index']:>3}. {label:<30} {c['count']:>5}  ({c['share']:.1%})  {bar}")


def _query_drift(args, state):
    from .lenses import drift

    events = drift.detect(state.snapshots) if len(state.snapshots) >= 2 else []

    data = {
        "snapshot_count": len(state.snapshots),
        "events": [
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
            for e in events
        ],
    }

    if args.json_output:
        _print_json(data)
        return

    if len(state.snapshots) < 2:
        print(f"Need >= 2 snapshots for drift. Current: {len(state.snapshots)}")
        print("Use 'convmap snapshot' to capture snapshots.")
        return

    print(f"\nDrift across {len(state.snapshots)} snapshots:\n")

    if not events:
        print("  No significant drift.")
        return

    for e in events:
        label_from = e["from_label"] or "?"
        label_to = e["to_label"] or "?"
        print(f"  [{e['magnitude']}] {label_from} -> {label_to}")
        d = e["diff"]
        print(f"    appeared: {d['appeared']}  disappeared: {d['disappeared']}")
        print(f"    moved: {len(d['moved'])}  grew: {len(d['grew'])}  shrunk: {len(d['shrunk'])}")


def _query_anomalies(args, state):
    from . import query

    time_range = _parse_time_range(args)
    results = query.anomalies(state, threshold=args.threshold, time_range=time_range)

    data = {
        "threshold": args.threshold,
        "count": len(results),
        "anomalies": [
            {"best_similarity": r["best_similarity"], "best_cluster": r["best_cluster"],
             "metadata": r["metadata"]}
            for r in results
        ],
    }

    if args.json_output:
        _print_json(data)
        return

    print(f"\nAnomalies (threshold={args.threshold}): {len(results)} found\n")

    for i, r in enumerate(results[:20]):
        meta = r["metadata"]
        conv_id = meta.get("id", "?")
        print(
            f"  {i + 1}. [{r['best_similarity']:.4f}] {conv_id}"
            f"  (nearest: cluster {r['best_cluster']})"
        )

    if len(results) > 20:
        print(f"  ... and {len(results) - 20} more")


def _query_segment(args, state):
    from . import query

    time_range = _parse_time_range(args)
    members = query.segment(state, args.index, time_range=time_range)

    label = None
    if args.index < len(state.core_clusters):
        label = state.core_clusters[args.index].label

    data = {
        "cluster_index": args.index,
        "label": label,
        "count": len(members),
        "members": [
            {"similarity": m["similarity"], "metadata": m["metadata"]}
            for m in members
        ],
    }

    if args.json_output:
        _print_json(data)
        return

    if not members:
        print(f"Cluster {args.index}: no members or invalid index.")
        return

    print(f"\nSegment: cluster {args.index}" + (f" ({label})" if label else ""))
    print(f"Members: {len(members)}\n")

    for i, m in enumerate(members[:20]):
        meta = m["metadata"]
        conv_id = meta.get("id", "?")
        print(f"  {i + 1}. [{m['similarity']:.4f}] {conv_id}")
        for k, v in meta.items():
            if k not in ("id", "n_chunks") and isinstance(v, str) and len(v) > 2:
                print(f"     {k}: {_truncate(v, 100)}")

    if len(members) > 20:
        print(f"  ... and {len(members) - 20} more")


def _query_compare(args, state):
    from . import query

    sa = _parse_timestamp(args.start_a)
    ea = _parse_timestamp(args.end_a)
    sb = _parse_timestamp(args.start_b)
    eb = _parse_timestamp(args.end_b)

    result = query.compare_windows(
        state,
        window_a=(sa, ea),
        window_b=(sb, eb),
    )

    if args.json_output:
        _print_json(result)
        return

    print(f"\nCompare windows:")
    print(f"  A [{args.start_a} -> {args.end_a}]: {result['window_a_count']} vectors")
    print(f"  B [{args.start_b} -> {args.end_b}]: {result['window_b_count']} vectors\n")

    if not result["cluster_shifts"]:
        print("  No shifts detected.")
        return

    print("  Shifts (by magnitude):\n")
    for s in result["cluster_shifts"]:
        label = s["label"] or f"cluster-{s['cluster_index']}"
        if s["delta"] > 0:
            arrow = "\u2191"
        elif s["delta"] < 0:
            arrow = "\u2193"
        else:
            arrow = "="
        print(
            f"    {label:<30} {s['share_a']:.1%} -> {s['share_b']:.1%}"
            f"  {arrow} {abs(s['delta']):.1%}"
        )


def _query_topology(args, state):
    from .lenses import topology

    edges = topology.adjacency(state)
    bridge_list = topology.bridges(state)
    iso = topology.isolated(state)
    dmap = topology.density_map(state)

    data = {
        "core_clusters": len(state.core_clusters),
        "adjacency": [
            {"i": e["i"], "j": e["j"], "similarity": e["similarity"],
             "label_i": e["label_i"], "label_j": e["label_j"]}
            for e in edges
        ],
        "isolated": [
            {"index": c["index"], "label": c["label"],
             "max_similarity": c["max_similarity"]}
            for c in iso
        ],
        "bridges_count": len(bridge_list),
        "density_map": dmap,
    }

    if args.json_output:
        _print_json(data)
        return

    print(f"\nTopology: {len(state.core_clusters)} core clusters\n")

    if edges:
        print(f"Adjacency ({len(edges)} edges):")
        for e in edges[:20]:
            li = e["label_i"] or f"cluster-{e['i']}"
            lj = e["label_j"] or f"cluster-{e['j']}"
            print(f"  {li} <-> {lj}  ({e['similarity']:.4f})")
        if len(edges) > 20:
            print(f"  ... and {len(edges) - 20} more")
        print()

    if iso:
        print(f"Isolated ({len(iso)}):")
        for c in iso:
            label = c["label"] or f"cluster-{c['index']}"
            print(f"  {label}  (max sim: {c['max_similarity']:.4f})")
        print()

    if bridge_list:
        print(f"Bridges ({len(bridge_list)}):")
        for b in bridge_list[:10]:
            conv_id = b["metadata"].get("id", "?")
            clusters_str = ", ".join(str(i) for i in b["cluster_indices"])
            print(f"  {conv_id} -> clusters [{clusters_str}]")
        if len(bridge_list) > 10:
            print(f"  ... and {len(bridge_list) - 10} more")
        print()

    if not edges and not iso and not bridge_list:
        print("  No topology data (need >= 2 core clusters).")


def _query_tda(args, state):
    from .lenses import tda

    source = args.source
    feat_list = tda.features(state, source=source)

    data = {
        "source": source,
        "features": feat_list[:50],
    }

    if args.json_output:
        _print_json(data)
        return

    print(f"\nTDA features (source={source}):\n")

    if not feat_list:
        print("  Not enough points for TDA analysis.")
        return

    for f in feat_list[:20]:
        print(f"  {f['type']:<12} rank={f['rank']}  lifetime={f['lifetime']:.4f}")

    if len(feat_list) > 20:
        print(f"\n  ... and {len(feat_list) - 20} more")


def _query_recluster(args, state):
    from .lenses import recluster as recluster_lens

    result = recluster_lens.recluster(
        state,
        epsilon=args.epsilon,
        mu=args.mu,
        beta=args.beta,
    )

    if args.json_output:
        _print_json(result)
        return

    p = result["params"]
    print(f"\nRe-cluster (epsilon={p['epsilon']}, mu={p['mu']}, beta={p['beta']})")
    print(f"Total vectors: {result['total_vectors']}\n")

    if not result["core_clusters"]:
        print("  No core clusters formed with these parameters.")
        if result["potential_clusters"]:
            print(f"  {len(result['potential_clusters'])} potential clusters.")
        print(f"  {result['outlier_count']} outliers.")
        return

    print(f"Core clusters ({len(result['core_clusters'])}):\n")
    for c in result["core_clusters"]:
        print(f"  {c['index']}. weight={c['weight']:.1f}  count={c['count']}  members={c['member_count']}")
        if c["top_members"]:
            for m in c["top_members"][:3]:
                meta = m["metadata"]
                conv_id = meta.get("id", "?")
                print(f"     [{m['similarity']:.4f}] {conv_id}")

    if result["potential_clusters"]:
        print(f"\nPotential clusters ({len(result['potential_clusters'])}):")
        for p in result["potential_clusters"][:10]:
            print(f"  weight={p['weight']:.1f}  count={p['count']}")

    print(f"\nOutliers: {result['outlier_count']}")


def _query_sweep(args, state):
    from .lenses import recluster as recluster_lens

    epsilon_values = args.epsilons if hasattr(args, "epsilons") and args.epsilons else None
    results = recluster_lens.sweep(state, epsilon_values=epsilon_values, mu=args.mu)

    data = {"mu": args.mu, "results": results}

    if args.json_output:
        _print_json(data)
        return

    print(f"\nSweep (mu={args.mu}):\n")
    print(f"  {'Epsilon':>8}  {'Min Sim':>8}  {'Core':>6}  {'Potential':>10}  {'Outliers':>8}")
    print(f"  {'─' * 8}  {'─' * 8}  {'─' * 6}  {'─' * 10}  {'─' * 8}")

    for r in results:
        print(
            f"  {r['epsilon']:>8.2f}  {r['min_similarity']:>8.2f}"
            f"  {r['core_clusters']:>6}  {r['potential_clusters']:>10}"
            f"  {r['outlier_count']:>8}"
        )


# ── Report ──────────────────────────────────────────────────────────


def _cmd_report(args):
    from . import persistence, query

    map_path = Path(args.map)
    if not (map_path / "meta.json").exists():
        print(f"No map at {map_path}. Run 'convmap import' first.", file=sys.stderr)
        sys.exit(1)

    engine = persistence.load(map_path)
    state = engine.state

    data = query.report(state, anomaly_threshold=args.anomaly_threshold)

    import json

    if args.report_format == "markdown":
        output = query.report_markdown(data)
    else:
        output = json.dumps(data, indent=2)

    if args.output:
        out_path = Path(args.output)
        out_path.write_text(output, encoding="utf-8")
        print(f"Report written to {out_path}")
    else:
        print(output)


# ── Status / Snapshot ───────────────────────────────────────────────


def _cmd_status(args):
    from . import persistence

    map_path = Path(args.map)
    if not (map_path / "meta.json").exists():
        print(f"No map at {map_path}.")
        return

    engine = persistence.load(map_path)
    s = engine.summary

    print(f"\nMap: {map_path}")
    print(f"  Dimensions:          {engine.dimensions}")
    print(f"  Core clusters:       {s['core_clusters']}")
    print(f"  Potential clusters:   {s['potential_clusters']}")
    print(f"  Outlier buffer:      {s['outlier_buffer']}")
    print(f"  Recent vectors:      {s['recent_vectors']}")
    print(f"  Snapshots:           {s['snapshots']}")
    print(f"  Total ingested:      {s['total_ingested']}")
    print(f"  Config:")
    print(f"    epsilon:             {engine.epsilon}")
    print(f"    mu:                  {engine.mu}")
    print(f"    beta:                {engine.beta}")
    print(f"    decay:               {engine.decay}")
    print(f"    maintenance:         {engine.maintenance_interval}")
    print(f"    max_recent:          {engine.max_recent}")


def _cmd_snapshot(args):
    from . import persistence

    map_path = Path(args.map)
    if not (map_path / "meta.json").exists():
        print(f"No map at {map_path}.", file=sys.stderr)
        sys.exit(1)

    engine = persistence.load(map_path)
    snap = engine.snapshot(label=args.label)
    persistence.save(engine, map_path)

    print(f"Snapshot captured:")
    print(f"  Label:      {snap.label or '(none)'}")
    print(f"  Core:       {snap.n_core}")
    print(f"  Potential:   {snap.n_potential}")
    print(f"  Outliers:   {snap.n_outliers}")
    print(f"  Total:      {len(engine.snapshots)} snapshots")


def _cmd_label(args):
    from . import persistence

    map_path = Path(args.map)
    if not (map_path / "meta.json").exists():
        print(f"No map at {map_path}.", file=sys.stderr)
        sys.exit(1)

    engine = persistence.load(map_path)

    if args.index >= len(engine.core_clusters):
        print(
            f"Invalid index {args.index}. Map has {len(engine.core_clusters)} core clusters.",
            file=sys.stderr,
        )
        sys.exit(1)

    engine.core_clusters[args.index].label = args.name
    persistence.save(engine, map_path)
    print(f"Cluster {args.index} labeled: {args.name}")


# ── Helpers ─────────────────────────────────────────────────────────


def _parse_timestamp(value: str) -> float:
    """Parse a timestamp string — epoch float or ISO 8601."""
    try:
        return float(value)
    except ValueError:
        pass
    from datetime import datetime
    return datetime.fromisoformat(value).timestamp()


def _parse_time_range(args) -> tuple[float, float] | None:
    start = getattr(args, "time_start", None)
    end = getattr(args, "time_end", None)
    if start is not None and end is not None:
        return (_parse_timestamp(start), _parse_timestamp(end))
    return None


def _truncate(s: str, max_len: int) -> str:
    if len(s) <= max_len:
        return s
    return s[: max_len - 3] + "..."


if __name__ == "__main__":
    main()
