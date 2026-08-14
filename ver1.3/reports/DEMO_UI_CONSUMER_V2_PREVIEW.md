# Consumer Demo UI v2 Preview

## Status

- Branch: `free-assessment-integration-v1`
- Preview file: `debug_ui/consumer_v2.html`
- Product-first local launcher: `scripts/consumer_demo.py`
- Existing `debug_ui/index.html`: unchanged
- Scoring/backend/API behavior: unchanged
- Hugging Face default page: unchanged until the preview is explicitly promoted

## Design correction

The first consumer preview was too close to a generic SaaS card dashboard. It has been replaced rather than incrementally patched.

The current preview intentionally follows the visual language of the owner's personal website (`kayui-gavo.github.io`):

- warm ivory / paper background;
- deep navy typography;
- restrained dusty-rose and muted-gold accents;
- serif display type paired with a neutral sans-serif body face;
- thin editorial rules instead of dense cards and pills;
- large typographic hierarchy and generous whitespace;
- low-chrome navigation;
- research diagnostics visually demoted behind disclosure.

The page is still a consumer practice product, not a clone of the personal profile page. The shared part is the design system and editorial rhythm.

## Consumer information architecture

The normal learner flow is now:

1. choose `音読練習` or `自由発話`;
2. read/listen to the target when one exists;
3. make one recording;
4. confirm ASR text only when free-speaking needs it;
5. see the practice reference score, available dimensions, and concrete feedback;
6. expand technical analysis only when wanted.

The score result deliberately uses a large editorial numeral instead of a gamified donut/ring. Dimension evidence is rendered as compact rows rather than a grid of KPI cards.

## Missing-cache startup fix

The legacy `scripts/debug_ui.py --public-demo` defaults point to `cache/ramen_kudasai`, which is a generated local cache and is not committed to the repository.

The repository already includes a complete pre-generated reference under:

```text
assets/reference_cache/ramen_kudasai_aivis.{json,npz,ref.wav}
```

`scripts/consumer_demo.py` now uses these bundled assets by default and redirects `/` to the consumer preview. It keeps explicit `--cache` / `--wav` overrides available.

Recommended local start command:

```bash
python scripts/consumer_demo.py
```

Then open:

```text
http://127.0.0.1:8765/
```

## Safety constraints

The preview deliberately does **not**:

- change ProductScore v2;
- enable ProductScore v3;
- invent a dimension value when the API marks it unavailable;
- treat unavailable evidence as zero;
- average research/vendor evidence;
- hide ASR confirmation in free-speaking mode;
- remove the existing research UI;
- introduce a paid API.

## Promotion gate

Do not replace `debug_ui/index.html` or the Hugging Face Space default page until the preview has been manually checked for:

- local zero-setup startup through `scripts/consumer_demo.py`;
- microphone permission and recording;
- fixed-reading evaluation;
- free-speaking ASR confirmation;
- upload flow;
- missing-dimension rendering;
- desktop and mobile layout;
- all four language shells;
- public-space cold-start/failure behavior.

If the preview is worse, revert the consumer-specific commits or delete the preview/launcher. `main` and the existing research UI remain untouched.
