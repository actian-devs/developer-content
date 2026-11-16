"""Command-line interface.

    chroma2actian list       --source ...                      list collections in the source
    chroma2actian preflight  --source ... --collection NAME    read-only scan, no writes
    chroma2actian migrate    --source ... --collection NAME    streaming copy, then validate
    chroma2actian validate   --source ... --collection NAME    verify an existing migration

Exit codes: 0 ok | 1 check/runtime failure | 2 usage or precondition | 3 interrupted (resume with --resume)
"""

from __future__ import annotations

import argparse
import json
import logging
import signal
import sys
import threading
import uuid
from pathlib import Path
from typing import Any, Callable, Sequence

from . import __version__
from .config import ActianConfig, ChromaConfig, env, env_bool
from .errors import EXIT_CHECK_FAILED, EXIT_OK, EXIT_USAGE, MigrationError, UsageError
from .pipeline import MigrationOptions, migrate_collection
from .preflight import PreflightOptions, run_preflight
from .source import ChromaSource
from .target import ActianTarget
from .transform import DEFAULT_UUID_NAMESPACE, IdStrategy, TransformOptions
from .validate import ValidateOptions, run_validation

log = logging.getLogger("chroma2actian")

# Seams for tests: replace to inject fakes.
source_factory: Callable[[ChromaConfig], ChromaSource] = ChromaSource
target_factory: Callable[[ActianConfig], ActianTarget] = ActianTarget


# ------------------------------------------------------------------ logging
class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        return json.dumps({"ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"), "level": record.levelname,
                           "logger": record.name, "msg": record.getMessage()})


def _setup_logging(level: str, fmt: str) -> None:
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(_JsonFormatter() if fmt == "json" else
                         logging.Formatter("%(asctime)s %(levelname)-7s %(message)s", "%H:%M:%S"))
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level.upper())
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("chromadb").setLevel(logging.WARNING)


# ------------------------------------------------------------------ parsing
def _bool_opt(p: argparse.ArgumentParser, flag: str, dest: str, default: Any, help_: str) -> None:
    p.add_argument(flag, dest=dest, action=argparse.BooleanOptionalAction, default=default, help=help_)


def _add_source_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("source (Chroma)")
    g.add_argument("--source", required=True, choices=("local", "http", "cloud"),
                   help="local = on-disk PersistentClient | http = self-hosted Chroma server | cloud = Chroma Cloud")
    g.add_argument("--chroma-path", default=env("CHROMA_PATH"), help="[local] directory of the persistent store (env CHROMA_PATH)")
    g.add_argument("--chroma-host", default=env("CHROMA_HOST", "localhost"), help="[http] server host (env CHROMA_HOST)")
    g.add_argument("--chroma-port", type=int, default=int(env("CHROMA_PORT", "8000") or 8000), help="[http] server port (env CHROMA_PORT)")
    _bool_opt(g, "--chroma-ssl", "chroma_ssl", env_bool("CHROMA_SSL"), "[http] use HTTPS (env CHROMA_SSL)")
    g.add_argument("--chroma-token", default=env("CHROMA_TOKEN"), help="[http] bearer token, sent as 'Authorization: Bearer ...' (env CHROMA_TOKEN)")
    g.add_argument("--chroma-header", action="append", default=[], metavar="'Name: value'", help="[http] extra request header (repeatable)")
    g.add_argument("--chroma-api-key", default=env("CHROMA_API_KEY"), help="[cloud] API key (prefer env CHROMA_API_KEY)")
    g.add_argument("--chroma-tenant", default=env("CHROMA_TENANT"), help="tenant (required for cloud; env CHROMA_TENANT)")
    g.add_argument("--chroma-database", default=env("CHROMA_DATABASE"), help="database (required for cloud; env CHROMA_DATABASE)")
    g.add_argument("--chroma-cloud-host", default=env("CHROMA_CLOUD_HOST", "api.trychroma.com"), help="[cloud] API host")
    g.add_argument("--page-size", type=int, default=0, help="records per Chroma read (default 1000 local/http, 100 cloud; shrinks automatically on errors)")


def _add_target_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("target (Actian VectorAI DB)")
    g.add_argument("--actian-url", default=None, help="gRPC host:port (env ACTIAN_VECTORAI_URL, default localhost:6574)")
    g.add_argument("--actian-token", default=env("ACTIAN_VECTORAI_ACCESS_TOKEN") or env("ACTIAN_VECTORAI_API_KEY"),
                   help="access token (env ACTIAN_VECTORAI_ACCESS_TOKEN)")
    g.add_argument("--actian-tls", dest="actian_tls", action=argparse.BooleanOptionalAction, default=None, help="enable TLS")
    g.add_argument("--actian-tls-ca-cert", default=None, help="CA certificate file for TLS")
    g.add_argument("--actian-allow-insecure", action="store_true", default=None, help="allow credentials over plaintext to a non-loopback host")
    g.add_argument("--actian-timeout", type=float, default=None, help="per-RPC timeout seconds (default 30)")
    g.add_argument("--actian-max-retries", type=int, default=None, help="SDK-level retries per RPC (default 3)")
    g.add_argument("--actian-pool-size", type=int, default=None, help="gRPC connection pool size")
    g.add_argument("--wait-for-target", type=float, default=0.0, metavar="SECONDS", help="wait up to N seconds for the server to be healthy")


def _add_scope_args(p: argparse.ArgumentParser, *, target_name: bool = True) -> None:
    g = p.add_argument_group("collections")
    g.add_argument("--collection", action="append", default=[], metavar="NAME", help="source collection (repeatable)")
    g.add_argument("--all-collections", action="store_true", help="every collection in the source database")
    if target_name:
        g.add_argument("--target-collection", default=None, help="target name (single collection only; default: same name)")
        g.add_argument("--target-prefix", default="", help="prefix added to every target collection name")
    g.add_argument("--workdir", type=Path, default=Path("migration_work"), help="checkpoints, reports, dead-letter (one subfolder per collection)")


def _add_transform_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("data mapping (must be identical for preflight / migrate / validate)")
    g.add_argument("--id-strategy", choices=[s.value for s in IdStrategy], default="uuid5",
                   help="uuid5: every ID -> UUIDv5 | preserve: keep canonical UUIDs/ints, others -> UUIDv5 | numeric: strict ints")
    g.add_argument("--uuid-namespace", default=str(DEFAULT_UUID_NAMESPACE), help="UUIDv5 namespace (change only if you need a different mapping)")
    g.add_argument("--id-field", default="chroma_id", help="payload key that stores the original Chroma ID")
    g.add_argument("--document-field", default="document", help="payload key for Chroma documents")
    g.add_argument("--no-documents", dest="include_documents", action="store_false", help="do not copy documents")
    g.add_argument("--include-uris", action="store_true", help="also copy Chroma record URIs")
    g.add_argument("--uri-field", default="uri")
    g.add_argument("--distance", default="auto", choices=("auto", "cosine", "euclid", "dot"), help="override the metric derived from Chroma (l2->euclid, cosine->cosine, ip->dot)")
    g.add_argument("--allow-zero-vectors", action="store_true", help="keep all-zero vectors under cosine (otherwise rejected)")
    g.add_argument("--on-bad-record", choices=("fail", "skip"), default="fail", help="what to do with records that cannot be converted")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="chroma2actian", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    p.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    p.add_argument("--log-format", default="text", choices=("text", "json"))
    sub = p.add_subparsers(dest="command", required=True)

    ls = sub.add_parser("list", help="list source collections")
    _add_source_args(ls)

    pf = sub.add_parser("preflight", help="read-only scan; reports what will change or block")
    _add_source_args(pf); _add_scope_args(pf); _add_transform_args(pf)

    mg = sub.add_parser("migrate", help="copy collections (resumable, idempotent), then validate")
    _add_source_args(mg); _add_target_args(mg); _add_scope_args(mg); _add_transform_args(mg)
    g = mg.add_argument_group("migration")
    g.add_argument("--batch-size", type=int, default=256, help="points per Actian upsert")
    g.add_argument("--max-batch-mb", type=float, default=16.0, help="cap on the estimated size of one upsert")
    g.add_argument("--workers", type=int, default=4, help="parallel upserts per page")
    g.add_argument("--if-exists", choices=("fail", "reuse", "recreate"), default="fail", help="when the target collection exists")
    g.add_argument("--yes", action="store_true", help="confirm destructive actions (--if-exists recreate)")
    g.add_argument("--resume", action="store_true", help="continue from the checkpoint")
    g.add_argument("--fresh", action="store_true", help="discard any checkpoint and start over")
    g.add_argument("--dimension", type=int, default=None, help="only needed to create a target for an EMPTY source")
    g.add_argument("--hnsw-m", type=int, default=None)
    g.add_argument("--hnsw-ef-construct", type=int, default=None)
    g.add_argument("--no-flush", dest="flush", action="store_false", help="skip flush + ready check after loading")
    g.add_argument("--no-validate", dest="validate", action="store_false", help="skip the validation pass")
    _add_validate_args(mg)

    va = sub.add_parser("validate", help="verify a finished migration")
    _add_source_args(va); _add_target_args(va); _add_scope_args(va); _add_transform_args(va)
    _add_validate_args(va)
    return p


def _add_validate_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("validation")
    g.add_argument("--sample", type=int, default=500, help="records compared field-by-field")
    g.add_argument("--top-k", type=int, default=10)
    g.add_argument("--auto-queries", type=int, default=20, help="self-queries built from sampled vectors (when no golden file)")
    g.add_argument("--golden-queries", type=Path, default=None, help="JSONL of {'query_id','embedding'} produced with YOUR embedding model")
    g.add_argument("--min-mean-overlap", type=float, default=0.9)
    g.add_argument("--min-query-overlap", type=float, default=0.6)
    g.add_argument("--poll-timeout", type=float, default=300.0, help="seconds to wait for writes to become searchable")
    g.add_argument("--allow-extra-target-points", action="store_true", help="target may hold points that are not in the source")
    g.add_argument("--no-diagnose", dest="diagnose", action="store_false", help="don't keep ID sets in memory to list differences")
    g.add_argument("--report", type=Path, default=None, help="validation report path (default WORKDIR/<collection>/validation_report.json)")


# ------------------------------------------------------------------- wiring
def _chroma_config(a: argparse.Namespace) -> ChromaConfig:
    headers: dict[str, str] = {}
    for h in a.chroma_header:
        if ":" not in h:
            raise UsageError(f"--chroma-header expects 'Name: value', got {h!r}")
        k, v = h.split(":", 1)
        headers[k.strip()] = v.strip()
    cfg = ChromaConfig(mode=a.source, path=a.chroma_path, host=a.chroma_host, port=a.chroma_port, ssl=a.chroma_ssl,
                       headers=headers, token=a.chroma_token, api_key=a.chroma_api_key, tenant=a.chroma_tenant,
                       database=a.chroma_database, cloud_host=a.chroma_cloud_host, default_page_size=a.page_size)
    cfg.validate()
    return cfg


def _actian_config(a: argparse.Namespace) -> ActianConfig:
    return ActianConfig(url=a.actian_url, api_key=a.actian_token, tls=a.actian_tls, tls_ca_cert=a.actian_tls_ca_cert,
                        allow_insecure=a.actian_allow_insecure, timeout=a.actian_timeout,
                        max_retries=a.actian_max_retries, pool_size=a.actian_pool_size,
                        wait_for_target=a.wait_for_target)


def _transform(a: argparse.Namespace) -> TransformOptions:
    try:
        ns = uuid.UUID(a.uuid_namespace)
    except ValueError:
        raise UsageError(f"--uuid-namespace is not a UUID: {a.uuid_namespace!r}") from None
    return TransformOptions(id_strategy=IdStrategy(a.id_strategy), uuid_namespace=ns, id_field=a.id_field,
                            document_field=a.document_field, include_documents=a.include_documents,
                            include_uris=a.include_uris, uri_field=a.uri_field)


def _collections(a: argparse.Namespace, source: ChromaSource) -> list[tuple[str, str]]:
    if a.all_collections and a.collection:
        raise UsageError("use either --collection or --all-collections, not both")
    names = source.list_collections() if a.all_collections else list(dict.fromkeys(a.collection))
    if not names:
        raise UsageError("no collection selected: pass --collection NAME (repeatable) or --all-collections")
    tn = getattr(a, "target_collection", None)
    if tn and len(names) != 1:
        raise UsageError("--target-collection works with exactly one source collection; use --target-prefix instead")
    prefix = getattr(a, "target_prefix", "")
    return [(n, tn if tn else f"{prefix}{n}") for n in names]


def _validate_opts(a: argparse.Namespace, coll: str, target: str, topts: TransformOptions, workdir: Path) -> ValidateOptions:
    return ValidateOptions(collection=coll, target_collection=target, transform=topts, workdir=workdir,
                           distance_override=a.distance, sample=a.sample, top_k=a.top_k, auto_queries=a.auto_queries,
                           golden_queries=a.golden_queries, min_mean_overlap=a.min_mean_overlap,
                           min_query_overlap=a.min_query_overlap, poll_timeout=a.poll_timeout,
                           allow_extra_target_points=a.allow_extra_target_points, diagnose=a.diagnose,
                           report=a.report if (a.report and len(a.collection) <= 1 and not a.all_collections) else None,
                           allow_zero_vectors=a.allow_zero_vectors)


# ------------------------------------------------------------------ command
def run(argv: Sequence[str] | None = None, stop_event: threading.Event | None = None) -> int:
    args = build_parser().parse_args(argv)
    _setup_logging(args.log_level, args.log_format)
    stop = stop_event or threading.Event()
    try:
        source = source_factory(_chroma_config(args))
        source.connect()

        if args.command == "list":
            for n in source.list_collections():
                coll = source.get_collection(n)
                space, how = source.space(coll)
                print(f"{n}\tcount={source.count(coll)}\tspace={space} ({how})")
            return EXIT_OK

        topts = _transform(args)
        pairs = _collections(args, source)
        worst = EXIT_OK

        if args.command == "preflight":
            for coll, tgt in pairs:
                wd = args.workdir / coll
                rep = run_preflight(source, coll, PreflightOptions(
                    transform=topts, target_collection=tgt, distance_override=args.distance,
                    on_bad_record=args.on_bad_record, allow_zero_vectors=args.allow_zero_vectors, workdir=wd))
                if not rep["ok"]:
                    worst = EXIT_CHECK_FAILED
                print(f"preflight {coll}: {'OK' if rep['ok'] else 'BLOCKED'} -> {wd / 'preflight.json'}")
            return worst

        target = target_factory(_actian_config(args))
        target.connect()
        try:
            for coll, tgt in pairs:
                wd = args.workdir / coll
                try:
                    if args.command == "migrate":
                        migrate_collection(source, target, MigrationOptions(
                            collection=coll, target_collection=tgt, transform=topts, workdir=wd,
                            batch_size=args.batch_size, max_batch_bytes=int(args.max_batch_mb * 1024 * 1024),
                            workers=args.workers, if_exists=args.if_exists, assume_yes=args.yes, resume=args.resume,
                            fresh=args.fresh, on_bad_record=args.on_bad_record,
                            allow_zero_vectors=args.allow_zero_vectors, distance_override=args.distance,
                            dimension_override=args.dimension, hnsw_m=args.hnsw_m,
                            hnsw_ef_construct=args.hnsw_ef_construct, flush=args.flush, stop_event=stop))
                    if args.command == "validate" or (args.command == "migrate" and args.validate):
                        rep = run_validation(source, target, _validate_opts(args, coll, tgt, topts, wd))
                        print(f"validate {coll}: {'PASS' if rep['ok'] else 'FAIL'}")
                        if not rep["ok"]:
                            worst = max(worst, EXIT_CHECK_FAILED)
                except MigrationError as exc:
                    log.error("%s: %s", coll, exc)
                    worst = max(worst, exc.exit_code)
                    if exc.exit_code == 3:
                        break
        finally:
            target.close()
        return worst
    except MigrationError as exc:
        log.error("%s", exc)
        return exc.exit_code


def main_entry() -> None:
    stop = threading.Event()

    def _handler(signum: int, _frame: Any) -> None:
        if stop.is_set():
            raise KeyboardInterrupt
        log.warning("signal %d received: finishing the current page, saving the checkpoint, then exiting "
                    "(press Ctrl-C again to abort immediately)", signum)
        stop.set()

    signal.signal(signal.SIGINT, _handler)
    signal.signal(signal.SIGTERM, _handler)
    sys.exit(run(stop_event=stop))


if __name__ == "__main__":  # pragma: no cover
    main_entry()
