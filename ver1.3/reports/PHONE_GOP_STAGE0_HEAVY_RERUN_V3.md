# Phone-GOP Stage-0 corrected heavy rerun v3

Date: 2026-08-15
Branch: `free-assessment-integration-v1`
Product impact: **NONE**
Human-recording gate: **BLOCKED**

## Why this heavy rerun is now warranted

The previous heavy run was scientifically useful but exposed a target-side
reading bug in the JVS native anchor: surface G2P could analyze `明王` as a
personal-name reading rather than the intended `みょうおう`. The old repeated
JVS `9 / 77` negative-position pattern was therefore withdrawn.

The branch has since hardened the full Stage-0 path before spending any new
human recording time:

- reviewed JVS kana reading override with explicit provenance;
- resolved-reading phone generation only when a verified/manual reading is
  authoritative, while ordinary automatic targets preserve the frozen surface-
  context pyopenjtalk-plus phone contract;
- exact vectorized CTC fixed-sequence forward regression-tested against the
  scalar reference;
- ordinary clarity competitor inventory excludes pause/control labels and
  special morae `N` / `cl`;
- enumerated alignment-free SD features use ordinary substitutions for ordinary
  phones and canonical-plus-deletion only for `N` / `cl`;
- normalized SD/`Occ(i)` now uses a **position-specific** Japanese wildcard
  policy: ordinary segmental wildcard set for ordinary phones, canonical
  special-mora token plus graph deletion path for `N` / `cl`;
- hybrid criterion rows are construct-tagged so raw frame-local values for
  `N`/`cl` cannot be silently treated as ordinary clarity evidence;
- speaker-held-out criterion code now supports same-fold feature-family
  ablation, while explicitly refusing product selection/significance claims.

A GitHub Actions run at commit `484dc1a98258d7140cda3b546ecedd888e602c2b`
passed **329 tests / 6 existing warnings**. Later commits before this trigger
only kept preflight method/schema provenance synchronized with the new research
schemas.

## What this rerun must establish automatically

The heavy artifact is considered Stage-0-successful only if all of the
following remain true:

1. **JVS reviewed reading actually drives phone evidence.** The artifact must
   contain the reviewed `みょうおう` reading and must not silently return to
   surface-only G2P.
2. **The old identical `明王` false-alarm block disappears.** Any remaining
   native negative local margins are descriptive pressure-test evidence, not
   pronunciation errors.
3. **Special-mora construct separation survives model inference.** `N`/`cl`
   stay canonical targets, are absent from ordinary segmental competitor sets,
   and use construct-specific alignment-free policies.
4. **Controlled local edits remain directional.** `b→p`, `g→k`, `ts→s`,
   `sh→s`, `ch→sh`, `cl` deletion and `N` deletion must still produce the
   expected target-specific direction. No universal `LPR < 0` threshold is
   introduced.
5. **Official UME-JRF public learner-domain anchors remain usable.** The
   published `じぶつ / じんぶつ` pair remains an unlabeled real-L2 `N`
   presence/absence sanity contrast, not criterion truth.
6. **All three pinned CTC backbones remain reproducible.** Beatrice,
   DistilHuBERT dual CTC and WavLM dual CTC must emit finite target-specific
   evidence with immutable revision provenance.
7. **Normalized SD schema is current.** Artifacts must identify
   `paper_sd_norm_forward_japanese_position_mask_v3`, not the superseded global
   phone mask.
8. **Hybrid schema is current.** Artifacts must identify
   `hybrid_phone_criterion_feature_bundle_v2` and retain construct-role flags.
9. **CTC peakiness remains visible.** Entropy/top-1/blank-dominance diagnostics
   stay descriptive model properties, never pronunciation scores.
10. **No product promotion occurs.** `score_mapped=false`,
    `product_calibrated=false`, product `/100` unchanged, human GOP recording
    still blocked.

## Decision after this run

A green heavy rerun can promote the engineering stack only to:

**ready for labeled Japanese-L2 criterion validation**.

It does not authorize direct C-end `明瞭さ /100` mapping. The next scientific
question is whether ordinary-segmental logit/posterior/alignment-free/normalized
features, and their hybrid, improve speaker-held-out expert-label performance
beyond simpler feature families. Special morae must be evaluated separately
with duration/context evidence rather than being forced into the ordinary
clarity model.
