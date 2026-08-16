from pathlib import Path

path = Path(__file__).resolve().parent / "free_speech_listener_server_v10.py"
text = path.read_text(encoding="utf-8")
old = '''        completed_count = sum(bool(row["completed"]) for row in presentations)\n        constructs = self.rating_schema.get("validation_constructs") or {}\n'''
new = '''        def presentation_order(item: Mapping[str, Any]) -> tuple[int, str]:\n            variant = _text(item.get("presentation_variant"))\n            block = 0 if variant == "isolated" else 1\n            digest = hashlib.sha256(\n                f"{rater_id}|{variant}|{_text(item.get('presentation_id'))}".encode("utf-8")\n            ).hexdigest()\n            return block, digest\n\n        presentations.sort(key=presentation_order)\n        completed_count = sum(bool(row["completed"]) for row in presentations)\n        constructs = self.rating_schema.get("validation_constructs") or {}\n'''
if old not in text:
    raise SystemExit("listener ordering insertion marker not found")
text = text.replace(old, new, 1)
old2 = '''            "source_paths_exposed": False,\n        }\n'''
new2 = '''            "source_paths_exposed": False,\n            "presentation_order": "isolated_then_contextual_rater_hash_v1",\n        }\n'''
if old2 not in text:
    raise SystemExit("listener payload marker not found")
text = text.replace(old2, new2, 1)
path.write_text(text, encoding="utf-8")
