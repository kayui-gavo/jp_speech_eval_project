from pathlib import Path

path = Path(__file__).resolve().parent / "free_speech_listener_server_v10.py"
text = path.read_text(encoding="utf-8")

old_html = '<div id="ratingCard" class="card hidden"><div class="row"><span id="variant" class="muted"></span><span id="progress" class="progress"></span></div><div id="contextBox" class="context hidden"></div><audio id="audio" controls preload="metadata"></audio>'
new_html = '<div id="ratingCard" class="card hidden"><div class="row"><span id="variant" class="muted"></span><span id="progress" class="progress"></span></div><div id="contextBox" class="context hidden"></div><audio id="contextAudio" class="hidden" controls preload="metadata"></audio><audio id="audio" controls preload="metadata"></audio>'
if old_html not in text:
    raise SystemExit("listener context HTML marker not found")
text = text.replace(old_html, new_html, 1)

old_js = 'if(current.context_presented){$("contextBox").textContent=current.context_text||"文脈音声を確認してください。";$("contextBox").classList.remove("hidden")}else{$("contextBox").classList.add("hidden");$("contextBox").textContent=""}$("audio").src=`/api/audio/${encodeURIComponent(current.audio_asset_id)}`;$("audio").load();'
new_js = '''if(current.context_presented){
  $("contextBox").textContent=current.context_text||"会話の前の音声を聞いてから、評価対象の発話を確認してください。";
  $("contextBox").classList.remove("hidden");
  if(current.context_audio_asset_id){
    $("contextAudio").src=`/api/audio/${encodeURIComponent(current.context_audio_asset_id)}`;
    $("contextAudio").classList.remove("hidden");
    $("contextAudio").load();
  }else{
    $("contextAudio").pause();$("contextAudio").removeAttribute("src");$("contextAudio").load();$("contextAudio").classList.add("hidden");
  }
}else{
  $("contextBox").classList.add("hidden");$("contextBox").textContent="";
  $("contextAudio").pause();$("contextAudio").removeAttribute("src");$("contextAudio").load();$("contextAudio").classList.add("hidden");
}
$("audio").src=`/api/audio/${encodeURIComponent(current.audio_asset_id)}`;$("audio").load();'''
if old_js not in text:
    raise SystemExit("listener context JS marker not found")
text = text.replace(old_js, new_js, 1)

path.write_text(text, encoding="utf-8")
