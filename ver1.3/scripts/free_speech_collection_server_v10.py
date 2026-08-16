#!/usr/bin/env python3
"""Local-only browser recorder for the v10 free-speech collection plan.

The server reads the deterministic assignment CSV and only accepts WAV writes
for predeclared sample ids and relative paths.  It does not expose held/dev or
learner/native labels to the participant page, and it never asks for a response
transcript.

Default binding is 127.0.0.1.  This is a research collection utility, not a
public endpoint and not part of the Hugging Face Space.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path, PurePosixPath
from typing import Any, Mapping
from urllib.parse import parse_qs, unquote, urlparse


SERVER_SCHEMA = "free_speech_collection_server_v10"
MAX_WAV_BYTES = 32 * 1024 * 1024


HTML = r'''<!doctype html>
<html lang="ja">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>日本語発話データ収録</title>
<style>
:root{--bg:#f5f7f8;--card:#fff;--ink:#17212b;--muted:#667085;--line:#d7dee6;--accent:#087f8c;--danger:#b42318;--ok:#067647}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font-family:-apple-system,BlinkMacSystemFont,"Hiragino Sans","Yu Gothic",sans-serif;line-height:1.55}
main{max-width:760px;margin:0 auto;padding:28px 18px 60px}.card{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:20px;box-shadow:0 8px 24px rgba(16,24,40,.06);margin-bottom:16px}
h1{font-size:22px;margin:0 0 5px}.sub{color:var(--muted);font-size:13px}.row{display:flex;gap:10px;align-items:center;flex-wrap:wrap}input{font:inherit;border:1px solid var(--line);border-radius:9px;padding:9px 11px;min-width:180px}button{font:inherit;border:0;border-radius:9px;padding:10px 15px;cursor:pointer;background:var(--accent);color:#fff;font-weight:700}button.secondary{background:#eef2f5;color:var(--ink)}button.danger{background:var(--danger)}button:disabled{opacity:.45;cursor:not-allowed}
.prompt{font-size:24px;font-weight:750;margin:14px 0 8px}.instruction{color:var(--muted)}.pill{display:inline-block;border:1px solid var(--line);border-radius:999px;padding:3px 9px;font-size:12px;color:var(--muted);margin-right:6px}.progress{font-size:13px;color:var(--muted)}
.status{padding:9px 11px;border-radius:9px;background:#f8fafb;margin-top:12px}.status.ok{color:var(--ok);background:#ecfdf3}.status.bad{color:var(--danger);background:#fef3f2}audio{width:100%;margin-top:12px}.hidden{display:none}.notice{font-size:13px;color:var(--muted);margin-top:10px}code{font-family:ui-monospace,SFMono-Regular,Menlo,monospace}
</style>
</head>
<body><main>
<div class="card">
<h1>日本語発話データ収録</h1>
<div class="sub">研究用ローカル収録ツール。表示された質問に自然に答えてください。</div>
<div class="row" style="margin-top:16px">
<label>参加者ID <input id="speakerId" autocomplete="off" placeholder="例: HL01"></label>
<button id="loadBtn">開始</button>
</div>
<div id="loadStatus" class="status hidden"></div>
</div>

<div id="taskCard" class="card hidden">
<div class="row"><span id="taskMode" class="pill"></span><span id="durationBucket" class="pill"></span><span id="progress" class="progress"></span></div>
<div id="prompt" class="prompt"></div>
<div id="instruction" class="instruction"></div>
<div class="notice">文章を読み上げる必要はありません。質問への答えを、自分の言葉で自然に話してください。</div>
<div class="row" style="margin-top:18px">
<button id="recordBtn">● 録音開始</button>
<button id="stopBtn" class="danger" disabled>■ 録音停止</button>
<button id="saveBtn" class="secondary" disabled>保存して次へ</button>
<button id="redoBtn" class="secondary" disabled>録り直す</button>
</div>
<audio id="preview" controls></audio>
<div id="recordStatus" class="status">録音待ち</div>
</div>

<div id="doneCard" class="card hidden"><h1>収録完了</h1><div class="sub">この参加者IDに割り当てられた収録はすべて保存されています。</div></div>
</main>
<script>
const $=id=>document.getElementById(id);
let speakerPayload=null,current=null,stream=null,ctx=null,source=null,processor=null,chunks=[],recording=false,wavBlob=null,previewUrl=null;

function status(node,text,kind=""){node.textContent=text;node.className=`status ${kind}`;node.classList.remove("hidden")}
function nextTask(){
  if(!speakerPayload)return;
  const pending=speakerPayload.assignments.filter(x=>!x.recorded);
  if(!pending.length){$("taskCard").classList.add("hidden");$("doneCard").classList.remove("hidden");return;}
  current=pending[0];
  $("doneCard").classList.add("hidden");$("taskCard").classList.remove("hidden");
  $("taskMode").textContent=current.task_mode==="spontaneous"?"自由発話":"会話応答";
  $("durationBucket").textContent=current.planned_duration_bucket==="short"?"短め":"長め";
  $("progress").textContent=`${speakerPayload.recorded_count+1} / ${speakerPayload.assignment_count}`;
  $("prompt").textContent=current.context_text;
  $("instruction").textContent=current.collection_instruction||"自然に答えてください。";
  resetTake();status($("recordStatus"),"録音待ち");
}
async function loadSpeaker(){
  const id=$("speakerId").value.trim();if(!id)return;
  try{const r=await fetch(`/api/speaker/${encodeURIComponent(id)}`);const p=await r.json();if(!r.ok)throw new Error(p.error||"IDを確認してください");speakerPayload=p;status($("loadStatus"),`参加者ID ${p.speaker_id}：未収録 ${p.pending_count} 件`,"ok");nextTask();}
  catch(e){status($("loadStatus"),e.message,"bad")}
}
function resetTake(){wavBlob=null;chunks=[];$("saveBtn").disabled=true;$("redoBtn").disabled=true;if(previewUrl)URL.revokeObjectURL(previewUrl);previewUrl=null;$("preview").removeAttribute("src");$("preview").load();}
function mergeChunks(parts){const n=parts.reduce((a,b)=>a+b.length,0),out=new Float32Array(n);let o=0;for(const p of parts){out.set(p,o);o+=p.length}return out}
function encodeWav(samples,sampleRate){const b=new ArrayBuffer(44+samples.length*2),v=new DataView(b);const s=(o,t)=>{for(let i=0;i<t.length;i++)v.setUint8(o+i,t.charCodeAt(i))};s(0,"RIFF");v.setUint32(4,36+samples.length*2,true);s(8,"WAVE");s(12,"fmt ");v.setUint32(16,16,true);v.setUint16(20,1,true);v.setUint16(22,1,true);v.setUint32(24,sampleRate,true);v.setUint32(28,sampleRate*2,true);v.setUint16(32,2,true);v.setUint16(34,16,true);s(36,"data");v.setUint32(40,samples.length*2,true);let o=44;for(let i=0;i<samples.length;i++,o+=2){const x=Math.max(-1,Math.min(1,samples[i]));v.setInt16(o,x<0?x*32768:x*32767,true)}return new Blob([b],{type:"audio/wav"})}
async function startRecording(){
  resetTake();
  try{
    stream=await navigator.mediaDevices.getUserMedia({audio:{channelCount:1,echoCancellation:false,noiseSuppression:false,autoGainControl:false}});
    ctx=new (window.AudioContext||window.webkitAudioContext)();source=ctx.createMediaStreamSource(stream);processor=ctx.createScriptProcessor(4096,1,1);chunks=[];recording=true;
    processor.onaudioprocess=e=>{if(!recording)return;chunks.push(new Float32Array(e.inputBuffer.getChannelData(0)))};
    source.connect(processor);processor.connect(ctx.destination);$("recordBtn").disabled=true;$("stopBtn").disabled=false;status($("recordStatus"),"録音中…","ok");
  }catch(e){status($("recordStatus"),`マイクを開始できません: ${e.message}`,"bad")}
}
async function stopRecording(){
  if(!recording)return;recording=false;$("stopBtn").disabled=true;$("recordBtn").disabled=false;
  try{processor&&processor.disconnect();source&&source.disconnect();stream&&stream.getTracks().forEach(t=>t.stop());const rate=ctx?ctx.sampleRate:48000;ctx&&await ctx.close();wavBlob=encodeWav(mergeChunks(chunks),rate);previewUrl=URL.createObjectURL(wavBlob);$("preview").src=previewUrl;$("saveBtn").disabled=false;$("redoBtn").disabled=false;status($("recordStatus"),`録音完了 ${(wavBlob.size/1024).toFixed(0)} KB。再生して確認してください。`)}catch(e){status($("recordStatus"),e.message,"bad")}
  finally{stream=ctx=source=processor=null}
}
async function saveTake(){
  if(!current||!wavBlob)return;$("saveBtn").disabled=true;
  try{const r=await fetch(`/api/save/${encodeURIComponent(current.sample_id)}`,{method:"POST",headers:{"Content-Type":"audio/wav"},body:wavBlob});const p=await r.json();if(!r.ok)throw new Error(p.error||"保存失敗");const item=speakerPayload.assignments.find(x=>x.sample_id===current.sample_id);item.recorded=true;speakerPayload.recorded_count++;speakerPayload.pending_count--;status($("recordStatus"),"保存しました","ok");setTimeout(nextTask,350)}
  catch(e){$("saveBtn").disabled=false;status($("recordStatus"),e.message,"bad")}
}
$("loadBtn").onclick=loadSpeaker;$("speakerId").addEventListener("keydown",e=>{if(e.key==="Enter")loadSpeaker()});$("recordBtn").onclick=startRecording;$("stopBtn").onclick=stopRecording;$("saveBtn").onclick=saveTake;$("redoBtn").onclick=()=>{resetTake();status($("recordStatus"),"録り直しできます")};
</script></body></html>'''


def _text(value: Any) -> str:
    return str(value or "").strip()


def _safe_relpath(value: str) -> Path:
    raw = _text(value).replace("\\", "/")
    pure = PurePosixPath(raw)
    if not raw or pure.is_absolute() or any(part in {"", ".", ".."} for part in pure.parts):
        raise ValueError(f"unsafe audio_relpath: {value!r}")
    if pure.suffix.lower() != ".wav":
        raise ValueError(f"audio_relpath must end in .wav: {value!r}")
    return Path(*pure.parts)


def _read_assignments(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        rows = [dict(row) for row in csv.DictReader(handle)]
    if not rows:
        raise ValueError("assignment CSV is empty")
    return rows


@dataclass(frozen=True)
class Assignment:
    sample_id: str
    speaker_id: str
    audio_relpath: Path
    task_mode: str
    planned_duration_bucket: str
    context_text: str
    collection_instruction: str

    def participant_payload(self, *, recorded: bool) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "task_mode": self.task_mode,
            "planned_duration_bucket": self.planned_duration_bucket,
            "context_text": self.context_text,
            "collection_instruction": self.collection_instruction,
            "recorded": recorded,
        }


class CollectionStore:
    def __init__(self, assignments_csv: str | Path, audio_root: str | Path):
        self.assignments_csv = Path(assignments_csv)
        self.audio_root = Path(audio_root).resolve()
        self.audio_root.mkdir(parents=True, exist_ok=True)
        self._by_sample: dict[str, Assignment] = {}
        self._by_speaker: dict[str, list[Assignment]] = {}
        for row in _read_assignments(assignments_csv):
            sample_id = _text(row.get("sample_id"))
            speaker_id = _text(row.get("speaker_id"))
            if not sample_id or not speaker_id:
                raise ValueError("assignment row missing sample_id or speaker_id")
            if sample_id in self._by_sample:
                raise ValueError(f"duplicate assignment sample_id: {sample_id}")
            assignment = Assignment(
                sample_id=sample_id,
                speaker_id=speaker_id,
                audio_relpath=_safe_relpath(_text(row.get("audio_relpath"))),
                task_mode=_text(row.get("task_mode")),
                planned_duration_bucket=_text(row.get("planned_duration_bucket")),
                context_text=_text(row.get("context_text")),
                collection_instruction=_text(row.get("collection_instruction")),
            )
            target = (self.audio_root / assignment.audio_relpath).resolve()
            if self.audio_root not in target.parents:
                raise ValueError(f"assignment escapes audio root: {sample_id}")
            self._by_sample[sample_id] = assignment
            self._by_speaker.setdefault(speaker_id, []).append(assignment)

    def target_path(self, sample_id: str) -> Path:
        assignment = self._by_sample.get(sample_id)
        if assignment is None:
            raise KeyError(sample_id)
        target = (self.audio_root / assignment.audio_relpath).resolve()
        if self.audio_root not in target.parents:
            raise ValueError(f"resolved target escapes audio root: {sample_id}")
        return target

    def speaker_payload(self, speaker_id: str) -> dict[str, Any]:
        assignments = self._by_speaker.get(speaker_id)
        if not assignments:
            raise KeyError(speaker_id)
        items = [item.participant_payload(recorded=self.target_path(item.sample_id).is_file()) for item in assignments]
        recorded_count = sum(bool(item["recorded"]) for item in items)
        return {
            "schema": SERVER_SCHEMA,
            "speaker_id": speaker_id,
            "assignment_count": len(items),
            "recorded_count": recorded_count,
            "pending_count": len(items) - recorded_count,
            "assignments": items,
            "analysis_labels_exposed": False,
            "response_transcript_requested": False,
        }

    def save_wav(self, sample_id: str, payload: bytes, *, overwrite: bool = False) -> Path:
        if len(payload) < 44 or len(payload) > MAX_WAV_BYTES:
            raise ValueError("invalid WAV byte length")
        if payload[:4] != b"RIFF" or payload[8:12] != b"WAVE":
            raise ValueError("payload is not a RIFF/WAVE file")
        target = self.target_path(sample_id)
        if target.exists() and not overwrite:
            raise FileExistsError(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        fd, temp_name = tempfile.mkstemp(prefix=target.name + ".", suffix=".tmp", dir=target.parent)
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(payload)
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


class CollectionHandler(BaseHTTPRequestHandler):
    server_version = "FreeSpeechCollectionV10/1.0"

    @property
    def store(self) -> CollectionStore:
        return self.server.store  # type: ignore[attr-defined]

    def _json(self, status: int, payload: Mapping[str, Any]) -> None:
        raw = json.dumps(dict(payload), ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(raw)

    def _html(self) -> None:
        raw = HTML.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self._html()
            return
        prefix = "/api/speaker/"
        if parsed.path.startswith(prefix):
            speaker_id = unquote(parsed.path[len(prefix):]).strip()
            try:
                self._json(HTTPStatus.OK, self.store.speaker_payload(speaker_id))
            except KeyError:
                self._json(HTTPStatus.NOT_FOUND, {"error": "unknown speaker id"})
            return
        self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        prefix = "/api/save/"
        if not parsed.path.startswith(prefix):
            self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})
            return
        sample_id = unquote(parsed.path[len(prefix):]).strip()
        try:
            content_length = int(self.headers.get("Content-Length") or "0")
        except ValueError:
            self._json(HTTPStatus.BAD_REQUEST, {"error": "invalid content length"})
            return
        if content_length <= 0 or content_length > MAX_WAV_BYTES:
            self._json(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, {"error": "invalid WAV size"})
            return
        payload = self.rfile.read(content_length)
        overwrite = parse_qs(parsed.query).get("overwrite", ["0"])[0] == "1"
        try:
            target = self.store.save_wav(sample_id, payload, overwrite=overwrite)
        except KeyError:
            self._json(HTTPStatus.NOT_FOUND, {"error": "unknown sample id"})
        except FileExistsError:
            self._json(HTTPStatus.CONFLICT, {"error": "recording already exists; use the participant page to continue instead of overwriting it"})
        except ValueError as exc:
            self._json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
        else:
            self._json(HTTPStatus.OK, {"ok": True, "sample_id": sample_id, "bytes": len(payload), "saved_name": target.name})

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"[collection] {self.address_string()} - {fmt % args}")


def serve(assignments_csv: str | Path, audio_root: str | Path, *, host: str = "127.0.0.1", port: int = 8765) -> None:
    store = CollectionStore(assignments_csv, audio_root)
    server = ThreadingHTTPServer((host, port), CollectionHandler)
    server.store = store  # type: ignore[attr-defined]
    print(f"v10 local collection server: http://{host}:{port}")
    print(f"assignments: {Path(assignments_csv).resolve()}")
    print(f"audio root: {Path(audio_root).resolve()}")
    if host not in {"127.0.0.1", "localhost", "::1"}:
        print("WARNING: non-loopback binding exposes the collection page on the network; use only on a trusted LAN.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("assignments_csv")
    parser.add_argument("--audio-root", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    serve(args.assignments_csv, args.audio_root, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
