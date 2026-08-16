from pathlib import Path

path = Path(__file__).resolve().parents[2] / ".github" / "workflows" / "baseline-evolution-light-tests.yml"
text = path.read_text(encoding="utf-8")

old_paths = '''      - "ver1.3/tests/test_free_speech_listener_order_v10.py"\n      - "ver1.3/tests/test_karaoke_timeline.py"'''
new_paths = '''      - "ver1.3/tests/test_free_speech_listener_order_v10.py"\n      - "ver1.3/tests/test_free_speech_listener_context_audio_v10.py"\n      - "ver1.3/tests/test_karaoke_timeline.py"'''
if old_paths not in text:
    raise SystemExit("permanent CI path marker not found")
text = text.replace(old_paths, new_paths, 1)

old_run = '''            tests/test_free_speech_listener_order_v10.py \\\n            tests/test_karaoke_timeline.py \\'''
new_run = '''            tests/test_free_speech_listener_order_v10.py \\\n            tests/test_free_speech_listener_context_audio_v10.py \\\n            tests/test_karaoke_timeline.py \\'''
if old_run not in text:
    raise SystemExit("permanent CI command marker not found")
text = text.replace(old_run, new_run, 1)

path.write_text(text, encoding="utf-8")
