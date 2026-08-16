from pathlib import Path

p = Path('ver1.3/debug_ui/index.html')
s = p.read_text(encoding='utf-8')


def replace_once(old: str, new: str, label: str) -> None:
    global s
    if old not in s:
        raise SystemExit(f'missing anchor: {label}')
    s = s.replace(old, new, 1)

replace_once(
    'state.stream = await navigator.mediaDevices.getUserMedia({ audio: true });',
    'state.stream = await navigator.mediaDevices.getUserMedia({ audio: { channelCount: 1, echoCancellation: false, noiseSuppression: false, autoGainControl: false } });',
    'raw microphone constraints',
)

replace_once(
    '''        if (!payload.ok) throw new Error(payload.error || "Sample failed");\n        renderPayload(payload);''',
    '''        if (!payload.ok) throw new Error(payload.error || "Sample failed");\n        $("lastRecordingAudio").src = $("sampleAudio").src;\n        $("lastRecordingAudio").load();\n        $("playLastBtn").disabled = false;\n        renderPayload(payload);''',
    'sample replay source',
)

replace_once(
    '''      $("playLastBtn").onclick = () => {\n        if (state.lastRecordingUrl) $("lastRecordingAudio").play();\n      };''',
    '''      $("playLastBtn").onclick = () => {\n        if ($("lastRecordingAudio").src) $("lastRecordingAudio").play();\n      };''',
    'play current replay audio',
)

replace_once(
    '''      $("fileInput").onchange = async (event) => {\n        const file = event.target.files[0];\n        if (file) await evaluateBlob(file, file.name).catch((err) => alert(err.message));\n      };''',
    '''      $("fileInput").onchange = async (event) => {\n        const file = event.target.files[0];\n        if (!file) return;\n        if (state.lastRecordingUrl) URL.revokeObjectURL(state.lastRecordingUrl);\n        state.lastRecordingUrl = URL.createObjectURL(file);\n        $("lastRecordingAudio").src = state.lastRecordingUrl;\n        $("lastRecordingAudio").load();\n        $("playLastBtn").disabled = false;\n        await evaluateBlob(file, file.name).catch((err) => alert(err.message));\n      };''',
    'uploaded audio replay source',
)

p.write_text(s, encoding='utf-8')
