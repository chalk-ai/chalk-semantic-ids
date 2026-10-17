"""Semantic-ID demo server: encode a title, and recommend the next video from a history.

Stdlib HTTP plus numpy and onnxruntime, so the image stays small and the replica is
serving within a second of Python starting. Paths default to where `deploy_demo.py`
puts things in the image; override them with env vars to run it locally:

    SEMANTIC_ID_ENCODER_DIR=.cache/minilm-onnx SEMANTIC_ID_ARTIFACTS=artifacts \\
        python demo/server.py --port 8080
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, urlparse

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
sys.path.insert(0, os.environ.get("SEMANTIC_ID_SRC", "/app"))
from recommender import Recommender, Sparse  # noqa: E402
from semantic_id_runtime import SemanticIdEncoder  # noqa: E402

ARTIFACTS = pathlib.Path(os.environ.get("SEMANTIC_ID_ARTIFACTS", "/app"))
ENCODER_DIR = pathlib.Path(os.environ.get("SEMANTIC_ID_ENCODER_DIR", "/opt/minilm"))
INDEX_HTML = pathlib.Path(os.environ.get("SEMANTIC_ID_INDEX", HERE / "index.html"))
# Chalk's webfonts, when the deploy baked them in (they are licensed, so not in this repo).
FONTS_DIR = pathlib.Path(os.environ.get("SEMANTIC_ID_FONTS", "/app/fonts"))
FONT_TYPES = {".woff2": "font/woff2", ".ttf": "font/ttf"}
MAX_HISTORY = 5


class Demo:
    def __init__(self) -> None:
        started = time.perf_counter()
        self.encoder = SemanticIdEncoder(ENCODER_DIR, ARTIFACTS / "rqvae.npz")
        data = np.load(ARTIFACTS / "recommender.npz")
        eval_report = json.loads((ARTIFACTS / "recommender_eval.json").read_text())
        self.titles: list[str] = json.loads((ARTIFACTS / "recommender_titles.json").read_text())
        self.lower_titles = [title.lower() for title in self.titles]
        self.video_ids = data["video_ids"]
        self.row_of = {int(v): i for i, v in enumerate(self.video_ids)}
        self.popularity = data["popularity"]
        self.model = Recommender(
            codes=data["codes"],
            embeddings=data["embeddings"].astype(np.float32),
            popularity=self.popularity,
            level0=data["level0"],
            level01=Sparse(*(data[f"level01_{f}"] for f in ("keys", "pointers", "targets", "counts"))),
            items=Sparse(*(data[f"items_{f}"] for f in ("keys", "pointers", "targets", "counts"))),
            weights=eval_report["weights"],
        )
        self.eval_report = eval_report
        # 72 MicroLens videos have an empty title; they all share one embedding and say
        # nothing to a viewer, so they are never suggested.
        self.blank = np.array([not title.strip() for title in self.titles])
        # Popular, non-blank titles make better "surprise me" picks than the long tail.
        order = np.argsort(-self.popularity)
        self.sample_pool = [int(i) for i in order[:3000] if self.titles[i].strip()]
        self.rng = np.random.default_rng()
        print(f"demo: loaded {len(self.titles)} videos in {time.perf_counter() - started:.2f}s", flush=True)

    def video(self, row: int) -> dict[str, Any]:
        return {
            "video_id": int(self.video_ids[row]),
            "title": self.titles[row],
            "semantic_id": [int(c) for c in self.model.codes[row]],
            "views": int(self.popularity[row]),
        }

    def search(self, query: str, limit: int) -> list[dict[str, Any]]:
        query = query.strip().lower()
        if not query:
            rows = self.rng.choice(self.sample_pool, size=limit, replace=False)
            return [self.video(int(r)) for r in rows]
        matches = [i for i, title in enumerate(self.lower_titles) if query in title]
        matches.sort(key=lambda i: -self.popularity[i])
        return [self.video(i) for i in matches[:limit]]

    def encode(self, title: str) -> dict[str, Any]:
        started = time.perf_counter()
        vector = self.encoder.embed([title])
        codes = self.encoder.quantize(vector)[0]
        elapsed_ms = 1000 * (time.perf_counter() - started)
        cosine = self.model.embeddings @ vector[0]
        neighbours: dict[str, list[dict[str, Any]]] = {}
        for depth in (1, 2):
            mask = (self.model.codes[:, :depth] == codes[:depth]).all(axis=1)
            rows = np.flatnonzero(mask)
            rows = rows[np.argsort(-cosine[rows])][:6]
            neighbours[str(depth)] = [{**self.video(int(r)), "cosine": round(float(cosine[r]), 3)} for r in rows]
        counts = {
            str(depth): int((self.model.codes[:, :depth] == codes[:depth]).all(axis=1).sum()) for depth in (1, 2, 3)
        }
        return {
            "title": title,
            "semantic_id": [int(c) for c in codes],
            "embedding": [round(float(x), 4) for x in vector[0]],
            "encode_ms": round(elapsed_ms, 1),
            "prefix_counts": counts,
            "neighbours": neighbours,
        }

    def recommend(self, history: list[dict[str, Any]], limit: int, mode: str) -> dict[str, Any]:
        started = time.perf_counter()
        codes, vectors, items, resolved = [], [], [], []
        for entry in history[:MAX_HISTORY]:
            if "video_id" in entry and int(entry["video_id"]) in self.row_of:
                row = self.row_of[int(entry["video_id"])]
                codes.append(self.model.codes[row])
                vectors.append(self.model.embeddings[row])
                items.append(row)
                resolved.append({**self.video(row), "new": False})
            elif str(entry.get("title", "")).strip():
                # A title the catalog has never seen: it gets a semantic ID right here,
                # and the prefix terms treat it like any other history entry.
                title = str(entry["title"]).strip()
                vector = self.encoder.embed([title])
                code = self.encoder.quantize(vector)[0]
                codes.append(code)
                vectors.append(vector[0])
                items.append(None)
                resolved.append(
                    {"video_id": None, "title": title, "semantic_id": [int(c) for c in code], "views": 0, "new": True}
                )
        if not codes:
            raise ValueError("history is empty")

        parts = self.model.parts(np.array(codes), np.array(vectors), items)
        if mode == "semantic":
            score = self.model.semantic_score(parts)
        elif mode == "content":
            score = parts["cosine"].copy()
        else:
            score = self.model.hybrid_score(parts)
        score[self.blank] = -np.inf
        for row in items:
            if row is not None:
                score[row] = -np.inf
        top = np.argsort(-score)[:limit]
        top_tokens = np.argsort(-parts["p0"])[:5]
        results = []
        for row in top:
            code = self.model.codes[row]
            shared = [
                {
                    "with": h["title"],
                    "depth": int(next((d for d in (3, 2, 1) if list(code[:d]) == h["semantic_id"][:d]), 0)),
                }
                for h in resolved
            ]
            results.append(
                {
                    **self.video(int(row)),
                    "score": round(float(score[row]), 3),
                    "why": {
                        "p_first_token": round(float(parts["p0"][code[0]]), 4),
                        "cosine": round(float(parts["cosine"][row]), 3),
                        "followed_count": round(float(parts["item_counts"][row]), 2),
                        "shared_prefix": max(shared, key=lambda s: s["depth"]),
                    },
                }
            )
        return {
            "history": resolved,
            "predicted_first_tokens": [{"token": int(t), "p": round(float(parts["p0"][t]), 4)} for t in top_tokens],
            "results": results,
            "recommend_ms": round(1000 * (time.perf_counter() - started), 1),
        }


def make_handler(demo: Demo) -> type[BaseHTTPRequestHandler]:
    index = INDEX_HTML.read_bytes()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 - stdlib signature
            pass

        def _send(self, status: int, body: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, payload: Any, status: int = HTTPStatus.OK) -> None:
            self._send(status, json.dumps(payload, ensure_ascii=False).encode(), "application/json; charset=utf-8")

        def do_GET(self) -> None:
            url = urlparse(self.path)
            query = parse_qs(url.query)
            if url.path in ("/", "/index.html"):
                self._send(HTTPStatus.OK, index, "text/html; charset=utf-8")
            elif url.path.startswith("/fonts/"):
                # Only a bare filename with a known font extension, so the path cannot escape.
                name = url.path.removeprefix("/fonts/")
                font = FONTS_DIR / name
                if "/" not in name and font.suffix in FONT_TYPES and font.is_file():
                    self._send(HTTPStatus.OK, font.read_bytes(), FONT_TYPES[font.suffix])
                else:
                    self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)
            elif url.path == "/healthz":
                self._json({"ok": True})
            elif url.path == "/api/meta":
                self._json({"videos": len(demo.titles), "eval": demo.eval_report})
            elif url.path == "/api/search":
                self._json(demo.search(query.get("q", [""])[0], min(int(query.get("n", ["12"])[0]), 50)))
            else:
                self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)

        def do_POST(self) -> None:
            try:
                length = min(int(self.headers.get("Content-Length", "0")), 64_000)
                body = json.loads(self.rfile.read(length) or b"{}")
                if self.path == "/api/encode":
                    title = str(body.get("title", "")).strip()[:500]
                    if not title:
                        raise ValueError("title is empty")
                    self._json(demo.encode(title))
                elif self.path == "/api/recommend":
                    mode = str(body.get("mode", "hybrid"))
                    if mode not in ("hybrid", "semantic", "content"):
                        raise ValueError(f"unknown mode {mode!r}")
                    self._json(demo.recommend(list(body.get("history", [])), min(int(body.get("n", 10)), 30), mode))
                else:
                    self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)
            except (ValueError, TypeError, json.JSONDecodeError) as exc:
                self._json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)

    return Handler


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()
    demo = Demo()
    server = ThreadingHTTPServer(("0.0.0.0", args.port), make_handler(demo))
    print(f"demo: serving on :{args.port}", flush=True)
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
