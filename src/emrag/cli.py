"""Command-line interface: ``emrag ingest | ask | stats``."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from emrag.config import Settings
from emrag.container import build_service
from emrag.errors import EmragError
from emrag.logging_setup import configure_logging
from emrag.models import Answer
from emrag.security import redact


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="emrag", description="Enterprise multi-agent RAG assistant"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    ingest = sub.add_parser("ingest", help="index a file or directory of documents")
    ingest.add_argument("path", type=Path)

    ask = sub.add_parser("ask", help="ask a question against the index")
    ask.add_argument("question")
    ask.add_argument("--json", action="store_true", dest="as_json", help="emit JSON")
    ask.add_argument("--trace", action="store_true", help="include the agent execution trace")

    sub.add_parser("stats", help="print index statistics")
    return parser


def _render(answer: Answer, *, show_trace: bool) -> str:
    lines = [answer.text]
    if answer.citations:
        lines += ["", "Sources:"]
        for citation in answer.citations:
            where = f" > {citation.section}" if citation.section else ""
            lines.append(f"  [{citation.index}] {citation.source}{where}")
    lines += ["", f"Grounding score: {answer.grounding_score:.2f}"]
    if answer.removed_claims:
        lines.append(f"Removed unsupported claims: {answer.removed_claims}")
    if show_trace:
        lines += ["", "Trace:"]
        lines += [f"  {e.node:<9} {e.duration_ms:8.2f} ms  {e.detail}" for e in answer.trace]
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point. Returns a process exit code."""
    args = _parser().parse_args(argv)
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")
    try:
        settings = Settings()
        configure_logging(settings.log_level, json_output=settings.log_json)
        service = build_service(settings)

        if args.command == "ingest":
            print(service.ingest(args.path).model_dump_json(indent=2))
        elif args.command == "ask":
            answer = service.ask(args.question)
            if args.as_json:
                exclude = None if args.trace else {"trace"}
                print(json.dumps(json.loads(answer.model_dump_json(exclude=exclude)), indent=2))
            else:
                print(_render(answer, show_trace=args.trace))
        else:
            print(json.dumps(service.stats(), indent=2))
    except EmragError as exc:
        print(f"error: {redact(str(exc))}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
