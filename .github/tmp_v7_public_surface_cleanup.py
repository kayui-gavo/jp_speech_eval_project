from pathlib import Path

p = Path('ver1.3/debug_ui/index.html')
s = p.read_text(encoding='utf-8')

anchors = [
    ('<section>\n        <div class="panel-head">\n          <h2 data-i18n="speechRegion">', '<section class="debug-only-panel">\n        <div class="panel-head">\n          <h2 data-i18n="speechRegion">', 'speech region'),
    ('<section>\n        <div class="panel-head">\n          <h2 data-i18n="pitchContour">', '<section class="debug-only-panel">\n        <div class="panel-head">\n          <h2 data-i18n="pitchContour">', 'legacy pitch'),
]
for old, new, label in anchors:
    if old not in s:
        raise SystemExit(f'missing {label} anchor')
    s = s.replace(old, new, 1)
p.write_text(s, encoding='utf-8')
