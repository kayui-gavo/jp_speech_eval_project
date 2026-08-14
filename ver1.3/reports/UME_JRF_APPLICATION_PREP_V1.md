# UME-JRF IDR application preparation v1

Date: 2026-08-15
Purpose: remove administrative ambiguity before requesting any new human recordings
Submission performed by this repository: **NO**
Terms accepted by this repository/tooling: **NO**

## Current official distribution route

The archived NII Speech Resources Consortium page says that, as of its 2026-03-16 update, speech corpora are distributed through NII's Informatics Research Data Repository (IDR).

Current UME-JRF application page:

`https://www.nii.ac.jp/dsc/idr/speech/submit/UME-JRF.html`

Corpus DOI:

`https://doi.org/10.32130/src.UME-JRF`

The corpus is free of charge but limited to research use.

## What the current IDR form requires

The UME-JRF application page currently requires all of the following fields:

- address;
- university/company name;
- laboratory/department;
- responsible person's name;
- responsible person's position/title;
- responsible person's email address, normally an institutional personal address;
- intended use, maximum 256 Japanese characters;
- agreement to the speech-corpus pledge and the IDR dataset-provision service terms.

The page explicitly says the **responsible person must be the laboratory/department/group responsible person or a person equivalent to full-time staff; a student cannot be the responsible person**.

After provisional submission, IDR sends a confirmation email to the responsible person's address. The applicant must complete the confirmation step, then the IDR office reviews the application; the page says this may take several days.

Because acceptance of terms and selection of the responsible institutional person are human/legal-administrative actions, the codebase must not auto-submit the form or pretend those steps are complete.

## Proposed research-purpose text

The form limits the intended-use field to 256 characters. A concise purpose aligned with the actual Stage-0 work is:

> 第二言語としての日本語発音自動評価に関する研究に使用する。日本語学習者音声に対する専門家評定と音響特徴量・音素CTCモデルの出力との関係を分析し、分節音・特殊拍・韻律の自動評価手法の妥当性を検証する。データは研究目的に限って利用し、商用製品には組み込まない。

This text deliberately:

- says automatic Japanese L2 pronunciation evaluation;
- says expert ratings are the criterion;
- covers segmental/special-mora/prosody research without conflating them;
- states research-only use;
- explicitly states the corpus will not be embedded in a commercial product.

## What should be obtained after approval

After lawful acquisition, do **not** start model fitting immediately. First run:

`python scripts/probe_ume_jrf_layout.py <LOCAL_UME_JRF_ROOT>`

The probe should confirm the actual corpus tree and locate/hash corpus-internal documentation, especially:

- `Vol1/doc/FJcontent/description.txt`
- `Vol1/doc/FJlabel/description.txt`

Only after the real grading documentation has been read should a concrete parser be implemented. Unknown numeric columns must never be guessed to be pronunciation labels.

## Data governance inside this project

UME-JRF is intentionally treated as external research data:

- no corpus WAV copied into repository assets;
- no raw expert-rating files committed to a public code repository unless the license explicitly permits redistribution (current policy assumes **do not redistribute**);
- derived research reports must not contain identifiable speaker information beyond corpus pseudonymous IDs needed for reproducibility;
- product runtime must not depend on the local UME-JRF path;
- research artifacts must carry `research_only_license=true` and `commercial_product_use_allowed=false`;
- no product `/100` mapping is learned merely because UME-JRF has expert ratings.

## Administrative gate

Repository preparation: **READY**

Actual IDR application: **HUMAN ACTION REQUIRED**

Reason: a responsible institutional person must be named, and a human must agree to the pledge/service terms and complete email confirmation. No new pronunciation recording is needed for this gate.
