#!/usr/bin/env python3
"""Local blinded listener-rating UI for the v10 free-speech criterion study.

Inputs are the existing v3 listener-pack CSV and its separate private asset map.
The browser receives only the fields already allowed by the blinded pack. Source
paths remain server-side. Responses are atomically stored one JSON document per
presentation and can be exported back into the existing v3 rating CSV schema.

This is a local research utility, not a public product endpoint.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path, PurePosixPath
from typing import Any, Mapping
from urllib.parse import unquote, urlparse

from validate_consumer_ratings_v3 import CONSTRUCT_TO_FIELD, validate_rating_file


SERVER_SCHEMA = "free_speech_listener_server_v10"
MAX_AUDIO_BYTES = 64 * 1024 * 1024
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RATING_SCHEMA = ROOT / "data" / "human_eval" / "consumer_rating_schema_v3.json"


HTML = r'''<!doctype html>
<html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>日本語発話 評価</title>
<style>
:root{--bg:#f5f7f8;--card:#fff;--ink:#182230;--muted:#667085;--line:#d6dde5;--accent:#087f8c;--ok:#067647;--bad:#b42318}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font-family:-apple-system,BlinkMacSystemFont,"Hiragino Sans","Yu Gothic",sans-serif;line-height:1.55}main{max-width:820px;margin:0 auto;padding:26px 18px 60px}.card{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:20px;margin-bottom:16px;box-shadow:0 8px 24px rgba(16,24,40,.06)}h1{font-size:22px;margin:0 0 5px}.sub,.muted{color:var(--muted);font-size:13px}.row{display:flex;gap:10px;align-items:center;flex-wrap:wrap}input{font:inherit;border:1px solid var(--line);border-radius:9px;padding:9px 11px}button{font:inherit;border:0;border-radius:9px;padding:10px 15px;background:var(--accent);color:white;font-weight:700;cursor:pointer}button.secondary{background:#eef2f5;color:var(--ink)}button:disabled{opacity:.45;cursor:not-allowed}.hidden{display:none}.status{margin-top:10px;padding:9px 11px;border-radius:9px;background:#f8fafb}.status.ok{color:var(--ok);background:#ecfdf3}.status.bad{color:var(--bad);background:#fef3f2}audio{width:100%;margin:14px 0}.context{border-left:3px solid var(--accent);padding:10px 12px;background:#f8fafb;margin:12px 0}.question{border-top:1px solid var(--line);padding:16px 0 4px}.question:first-child{border-top:0}.question-title{font-weight:700}.scale{display:grid;grid-template-columns:repeat(7,1fr);gap:6px;margin-top:10px}.scale label{border:1px solid var(--line);border-radius:9px;padding:9px 4px;text-align:center;cursor:pointer}.scale input{min-width:0;margin:0 0 3px}.anchors{display:flex;justify-content:space-between;gap:15px;color:var(--muted);font-size:12px;margin-top:6px}.yn{display:flex;gap:8px;margin-top:8px}.yn label{border:1px solid var(--line);border-radius:9px;padding:8px 13px}.progress{font-size:13px;color:var(--muted)}
</style></head><body><main>
<div class="card"><h1>日本語発話の聞き取り評価</h1><div class="sub">匿名化された研究用評価です。画面に表示された観点だけを評価してください。</div><div class="row" style="margin-top:14px"><label>評価者ID <input id="rater" autocomplete="off" placeholder="例: r01"></label><button id="load">開始</button></div><div id="loadStatus" class="status hidden"></div></div>
<div id="ratingCard" class="card hidden"><div class="row"><span id="variant" class="muted"></span><span id="progress" class="progress"></span></div><div id="contextBox" class="context hidden"></div><audio id="audio" controls preload="metadata"></audio><div class="question"><div class="question-title">この音声は評価できますか。</div><div class="yn"><label><input type="radio" name="analyzable" value="yes"> はい</label><label><input type="radio" name="analyzable" value="no"> いいえ</label></div></div><div id="questions"></div><div class="row" style="margin-top:18px"><button id="save">保存して次へ</button><button id="replay" class="secondary">もう一度再生</button></div><div id="saveStatus" class="status">評価待ち</div></div>
<div id="done" class="card hidden"><h1>割り当て分は完了です</h1><div class="sub">ご協力ありがとうございました。</div></div>
<script>
const $=id=>document.getElementById(id);let payload=null,current=null;
function setStatus(el,text,kind=""){el.textContent=text;el.className=`status ${kind}`;el.classList.remove("hidden")}
function esc(s){return String(s??"").replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;","\"":"&quot;","'":"&#39;"}[c]))}
async function loadRater(){const id=$("rater").value.trim();if(!id)return;try{const r=await fetch(`/api/rater/${encodeURIComponent(id)}`);const p=await r.json();if(!r.ok)throw new Error(p.error||"IDを確認してください");payload=p;setStatus($("loadStatus"),`未評価 ${p.pending_count} / ${p.assignment_count}`,"ok");next()}catch(e){setStatus($("loadStatus"),e.message,"bad")}}
function ratingQuestion(construct){const meta=payload.constructs[construct]||{};let radios="";for(let i=1;i<=7;i++)radios+=`<label><input type="radio" name="${esc(construct)}" value="${i}"><br>${i}</label>`;return `<div class="question" data-construct="${esc(construct)}"><div class="question-title">${esc(meta.instruction_ja||construct)}</div><div class="scale">${radios}</div><div class="anchors"><span>1 ${esc(payload.scale["1"]||"")}</span><span>7 ${esc(payload.scale["7"]||"")}</span></div></div>`}
function clearForm(){document.querySelectorAll('input[type="radio"]').forEach(x=>x.checked=false);$("questions").classList.remove("hidden");setStatus($("saveStatus"),"評価待ち")}
function next(){if(!payload)return;const pending=payload.presentations.filter(x=>!x.completed);if(!pending.length){$("ratingCard").classList.add("hidden");$("done").classList.remove("hidden");return;}current=pending[0];$("done").classList.add("hidden");$("ratingCard").classList.remove("hidden");$("variant").textContent=current.presentation_variant==="contextual"?"会話文脈あり":"発話単体";$("progress").textContent=`${payload.completed_count+1} / ${payload.assignment_count}`;if(current.context_presented){$("contextBox").textContent=current.context_text||"文脈音声を確認してください。";$("contextBox").classList.remove("hidden")}else{$("contextBox").classList.add("hidden");$("contextBox").textContent=""}$("audio").src=`/api/audio/${encodeURIComponent(current.audio_asset_id)}`;$("audio").load();$("questions").innerHTML=current.assigned_constructs.map(ratingQuestion).join("");clearForm()}
function selected(name){return document.querySelector(`input[name="${CSS.escape(name)}"]:checked`)?.value||""}
function updateAnalyzable(){const yes=selected("analyzable")==="yes";$("questions").classList.toggle("hidden",!yes)}
async function save(){if(!current)return;const analyzable=selected("analyzable");if(!analyzable){setStatus($("saveStatus"),"まず評価可能か選んでください。","bad");return}const ratings={};if(analyzable==="yes"){for(const c of current.assigned_constructs){const v=selected(c);if(!v){setStatus($("saveStatus"),"すべての項目を評価してください。","bad");return}ratings[c]=Number(v)}}try{const r=await fetch(`/api/rate/${encodeURIComponent(current.presentation_id)}`,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({analyzable_yes_no:analyzable,ratings})});const p=await r.json();if(!r.ok)throw new Error(p.error||"保存できませんでした");const row=payload.presentations.find(x=>x.presentation_id===current.presentation_id);row.completed=true;payload.completed_count++;payload.pending_count--;setStatus($("saveStatus"),"保存しました","ok");setTimeout(next,250)}catch(e){setStatus($("saveStatus"),e.message,"bad")}}
$("load").onclick=loadRater;$("rater").addEventListener("keydown",e=>{if(e.key==="Enter")loadRater()});document.addEventListener("change",e=>{if(e.target?.name==="analyzable")updateAnalyzable()});$("save").onclick=save;$("replay").onclick=()=>{$("audio").currentTime=0;$("audio").play()};
</script></body></html>'''


def _text(value: Any) -> str:
    return str(value or "").strip()


def _read_csv(path: str | Path) -> tuple[list[dict[str, str]], list[str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return [dict(row) for row in reader], list(reader.fieldnames or [])


def _safe_relpath(value: str) -> Path:
    raw = _text(value).replace("\\", "/")
    pure = PurePosixPath(raw)
    if not raw or pure.is_absolute() or any(part in {"", ".", ".."} for part in pure.parts):
        raise ValueError(f"unsafe source path: {value!r}")
    return Path(*pure.parts)


def _response_name(presentation_id: str) -> str:
    return hashlib.sha256(presentation_id.encode("utf-8")).hexdigest() + ".json"


def _assigned(value: Any) -> list[str]:
    return [item.strip() for item in _text(value).replace(",", "|").split("|") if item.strip()]


@dataclass(frozen=True)
class Asset:
    asset_id: str
    source_path: Path


class ListenerStore:
    def __init__(
        self,
        listener_csv: str | Path,
        asset_map_csv: str | Path,
        audio_root: str | Path,
        responses_dir: str | Path,
        *,
        rating_schema_json: str | Path = DEFAULT_RATING_SCHEMA,
    ):
        self.listener_csv = Path(listener_csv)
        self.rows, self.fieldnames = _read_csv(listener_csv)
        self.audio_root = Path(audio_root).resolve()
        self.responses_dir = Path(responses_dir)
        self.responses_dir.mkdir(parents=True, exist_ok=True)
        self.rating_schema = json.loads(Path(rating_schema_json).read_text(encoding="utf-8"))
        self.by_presentation: dict[str, dict[str, str]] = {}
        self.by_rater: dict[str, list[dict[str, str]]] = {}
        for row in self.rows:
            presentation_id = _text(row.get("presentation_id"))
            rater_id = _text(row.get("rater_id"))
            if not presentation_id or not rater_id:
                raise ValueError("listener row missing presentation_id/rater_id")
            if presentation_id in self.by_presentation:
                raise ValueError(f"duplicate presentation_id: {presentation_id}")
            self.by_presentation[presentation_id] = row
            self.by_rater.setdefault(rater_id, []).append(row)

        self.assets: dict[str, Asset] = {}
        asset_rows, _ = _read_csv(asset_map_csv)
        for row in asset_rows:
            for id_field, path_field in (
                ("audio_asset_id", "source_audio_path"),
                ("context_audio_asset_id", "source_context_audio_path"),
            ):
                asset_id = _text(row.get(id_field))
                source = _text(row.get(path_field))
                if not asset_id or not source:
                    continue
                rel = _safe_relpath(source)
                target = (self.audio_root / rel).resolve()
                if self.audio_root not in target.parents:
                    raise ValueError(f"asset escapes audio root: {asset_id}")
                old = self.assets.get(asset_id)
                if old is not None and old.source_path != target:
                    raise ValueError(f"asset id maps to multiple paths: {asset_id}")
                self.assets[asset_id] = Asset(asset_id=asset_id, source_path=target)

    def response_path(self, presentation_id: str) -> Path:
        if presentation_id not in self.by_presentation:
            raise KeyError(presentation_id)
        return self.responses_dir / _response_name(presentation_id)

    def completed(self, presentation_id: str) -> bool:
        return self.response_path(presentation_id).is_file()

    def participant_payload(self, rater_id: str) -> dict[str, Any]:
        rows = self.by_rater.get(rater_id)
        if not rows:
            raise KeyError(rater_id)
        presentations = []
        for row in rows:
            assigned = _assigned(row.get("assigned_constructs"))
            presentations.append(
                {
                    "presentation_id": _text(row.get("presentation_id")),
                    "presentation_variant": _text(row.get("presentation_variant")),
                    "audio_asset_id": _text(row.get("audio_asset_id")),
                    "context_presented": _text(row.get("context_presented_yes_no")).lower() == "yes",
                    "context_type": _text(row.get("context_type")),
                    "context_text": _text(row.get("context_text")),
                    "context_audio_asset_id": _text(row.get("context_audio_asset_id")),
                    "assigned_constructs": assigned,
                    "completed": self.completed(_text(row.get("presentation_id"))),
                }
            )
        def presentation_order(item: Mapping[str, Any]) -> tuple[int, str]:
            variant = _text(item.get("presentation_variant"))
            block = 0 if variant == "isolated" else 1
            digest = hashlib.sha256(
                f"{rater_id}|{variant}|{_text(item.get('presentation_id'))}".encode("utf-8")
            ).hexdigest()
            return block, digest

        presentations.sort(key=presentation_order)
        completed_count = sum(bool(row["completed"]) for row in presentations)
        constructs = self.rating_schema.get("validation_constructs") or {}
        allowed_constructs = {
            key: {"instruction_ja": _text((constructs.get(key) or {}).get("instruction_ja"))}
            for key in CONSTRUCT_TO_FIELD
        }
        return {
            "schema": SERVER_SCHEMA,
            "rater_id": rater_id,
            "assignment_count": len(presentations),
            "completed_count": completed_count,
            "pending_count": len(presentations) - completed_count,
            "presentations": presentations,
            "constructs": allowed_constructs,
            "scale": dict(self.rating_schema.get("scale") or {}),
            "analysis_metadata_exposed": False,
            "target_transcript_exposed": False,
            "source_paths_exposed": False,
            "presentation_order": "isolated_then_contextual_rater_hash_v1",
        }

    def audio_path(self, asset_id: str) -> Path:
        asset = self.assets.get(asset_id)
        if asset is None:
            raise KeyError(asset_id)
        if not asset.source_path.is_file():
            raise FileNotFoundError(asset.source_path)
        return asset.source_path

    def save_response(self, presentation_id: str, payload: Mapping[str, Any]) -> Path:
        row = self.by_presentation.get(presentation_id)
        if row is None:
            raise KeyError(presentation_id)
        analyzable = _text(payload.get("analyzable_yes_no")).lower()
        if analyzable not in {"yes", "no"}:
            raise ValueError("analyzable_yes_no must be yes or no")
        ratings = payload.get("ratings") if isinstance(payload.get("ratings"), Mapping) else {}
        assigned = set(_assigned(row.get("assigned_constructs")))
        normalized: dict[str, int] = {}
        if analyzable == "yes":
            if set(ratings) != assigned:
                raise ValueError("ratings must contain exactly the assigned constructs")
            for construct in assigned:
                try:
                    value = int(ratings[construct])
                except (TypeError, ValueError):
                    raise ValueError(f"invalid rating for {construct}") from None
                if not 1 <= value <= 7:
                    raise ValueError(f"rating out of range for {construct}")
                normalized[construct] = value
        elif ratings:
            raise ValueError("unanalyzable presentation must not contain construct ratings")

        result = {
            "schema": SERVER_SCHEMA,
            "presentation_id": presentation_id,
            "rater_id": _text(row.get("rater_id")),
            "analyzable_yes_no": analyzable,
            "ratings": normalized,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        target = self.response_path(presentation_id)
        fd, temp_name = tempfile.mkstemp(prefix=target.name + ".", suffix=".tmp", dir=target.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(result, handle, ensure_ascii=False, indent=2, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_name, target)
        except Exception:
            try:
                os.unlink(temp_name)
            except FileNotFoundError:
                pass
            raise
        return target

    def export_completed(self, output_csv: str | Path, *, require_complete: bool = False) -> dict[str, Any]:
        output_rows: list[dict[str, Any]] = []
        missing: list[str] = []
        for template in self.rows:
            presentation_id = _text(template.get("presentation_id"))
            response_path = self.response_path(presentation_id)
            if not response_path.is_file():
                missing.append(presentation_id)
                continue
            response = json.loads(response_path.read_text(encoding="utf-8"))
            if _text(response.get("presentation_id")) != presentation_id:
                raise ValueError(f"response presentation mismatch: {presentation_id}")
            if _text(response.get("rater_id")) != _text(template.get("rater_id")):
                raise ValueError(f"response rater mismatch: {presentation_id}")
            row = dict(template)
            row["analyzable_yes_no"] = _text(response.get("analyzable_yes_no"))
            for field in CONSTRUCT_TO_FIELD.values():
                row[field] = ""
            for construct, value in (response.get("ratings") or {}).items():
                if construct not in CONSTRUCT_TO_FIELD:
                    raise ValueError(f"unknown response construct: {construct}")
                row[CONSTRUCT_TO_FIELD[construct]] = value
            row["timestamp"] = _text(response.get("timestamp"))
            output_rows.append(row)
        if require_complete and missing:
            raise ValueError(f"incomplete listener pack: {len(missing)} presentations missing")
        output = Path(output_csv)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.fieldnames)
            writer.writeheader()
            writer.writerows(output_rows)
        validation = validate_rating_file(output)
        if not validation.get("ok"):
            raise ValueError("exported ratings failed v3 validation: " + ";".join(validation.get("errors") or []))
        return {
            "schema": SERVER_SCHEMA,
            "assigned_presentation_count": len(self.rows),
            "completed_presentation_count": len(output_rows),
            "missing_presentation_count": len(missing),
            "missing_presentation_ids": missing,
            "output_csv": str(output),
            "validation": validation,
        }


class ListenerHandler(BaseHTTPRequestHandler):
    server_version = "FreeSpeechListenerV10/1.0"

    @property
    def store(self) -> ListenerStore:
        return self.server.store  # type: ignore[attr-defined]

    def _json(self, status: int, payload: Mapping[str, Any]) -> None:
        raw = json.dumps(dict(payload), ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/":
            raw = HTML.encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(raw)
            return
        if parsed.path.startswith("/api/rater/"):
            rater_id = unquote(parsed.path[len("/api/rater/"):]).strip()
            try:
                self._json(HTTPStatus.OK, self.store.participant_payload(rater_id))
            except KeyError:
                self._json(HTTPStatus.NOT_FOUND, {"error": "unknown rater id"})
            return
        if parsed.path.startswith("/api/audio/"):
            asset_id = unquote(parsed.path[len("/api/audio/"):]).strip()
            try:
                path = self.store.audio_path(asset_id)
            except KeyError:
                self._json(HTTPStatus.NOT_FOUND, {"error": "unknown audio asset"})
                return
            except FileNotFoundError:
                self._json(HTTPStatus.NOT_FOUND, {"error": "audio file missing"})
                return
            size = path.stat().st_size
            if size > MAX_AUDIO_BYTES:
                self._json(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, {"error": "audio file too large"})
                return
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "audio/wav")
            self.send_header("Content-Length", str(size))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            with path.open("rb") as handle:
                while True:
                    chunk = handle.read(65536)
                    if not chunk:
                        break
                    self.wfile.write(chunk)
            return
        self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if not parsed.path.startswith("/api/rate/"):
            self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})
            return
        presentation_id = unquote(parsed.path[len("/api/rate/"):]).strip()
        try:
            length = int(self.headers.get("Content-Length") or "0")
        except ValueError:
            self._json(HTTPStatus.BAD_REQUEST, {"error": "invalid content length"})
            return
        if length <= 0 or length > 65536:
            self._json(HTTPStatus.BAD_REQUEST, {"error": "invalid response size"})
            return
        try:
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            if not isinstance(payload, Mapping):
                raise ValueError("response must be a JSON object")
            self.store.save_response(presentation_id, payload)
        except KeyError:
            self._json(HTTPStatus.NOT_FOUND, {"error": "unknown presentation id"})
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            self._json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
        else:
            self._json(HTTPStatus.OK, {"ok": True, "presentation_id": presentation_id})

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"[listener] {self.address_string()} - {fmt % args}")


def serve(
    listener_csv: str | Path,
    asset_map_csv: str | Path,
    audio_root: str | Path,
    responses_dir: str | Path,
    *,
    rating_schema_json: str | Path = DEFAULT_RATING_SCHEMA,
    host: str = "127.0.0.1",
    port: int = 8766,
) -> None:
    store = ListenerStore(listener_csv, asset_map_csv, audio_root, responses_dir, rating_schema_json=rating_schema_json)
    server = ThreadingHTTPServer((host, port), ListenerHandler)
    server.store = store  # type: ignore[attr-defined]
    print(f"v10 local listener server: http://{host}:{port}")
    print(f"responses: {Path(responses_dir).resolve()}")
    if host not in {"127.0.0.1", "localhost", "::1"}:
        print("WARNING: non-loopback binding exposes the listener UI and audio on the network; use only on a trusted LAN.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    serve_parser = sub.add_parser("serve")
    serve_parser.add_argument("listener_csv")
    serve_parser.add_argument("asset_map_csv")
    serve_parser.add_argument("--audio-root", required=True)
    serve_parser.add_argument("--responses-dir", required=True)
    serve_parser.add_argument("--rating-schema", default=str(DEFAULT_RATING_SCHEMA))
    serve_parser.add_argument("--host", default="127.0.0.1")
    serve_parser.add_argument("--port", type=int, default=8766)

    export_parser = sub.add_parser("export")
    export_parser.add_argument("listener_csv")
    export_parser.add_argument("asset_map_csv")
    export_parser.add_argument("--audio-root", required=True)
    export_parser.add_argument("--responses-dir", required=True)
    export_parser.add_argument("--rating-schema", default=str(DEFAULT_RATING_SCHEMA))
    export_parser.add_argument("--out", required=True)
    export_parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()

    if args.command == "serve":
        serve(
            args.listener_csv,
            args.asset_map_csv,
            args.audio_root,
            args.responses_dir,
            rating_schema_json=args.rating_schema,
            host=args.host,
            port=args.port,
        )
        return 0

    store = ListenerStore(
        args.listener_csv,
        args.asset_map_csv,
        args.audio_root,
        args.responses_dir,
        rating_schema_json=args.rating_schema,
    )
    report = store.export_completed(args.out, require_complete=args.require_complete)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
