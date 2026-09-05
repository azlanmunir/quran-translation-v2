# Software v2 Release Notes

## Repository Decision

The maintained destination is the existing private
`azlanmunir/quran-translation-v2` repository. Its initial GitHub snapshot
(`8ff9baeda3f36209a1ceb102efcb9eb46b09e160`) predates the completed English/Urdu,
video, and short-form workflows. This update preserves its development history
and incorporates the separately developed media code and reviewed fixes.

The public legacy repository is left unchanged. Its reviewed snapshot
(`72091071fc9669972b0b08fb0a790796a8f3bfaa`) contains no automated tests, and
`output/quran-metadata.md` reports only 59/114 clearly identified Surahs and six
skipped pages (490, 491, 548, 549, 553, 554). Its PDF-based runner marks filtered
pages complete, and appends output separately from checkpointing. Those behaviors
do not provide a reliable completeness or interruption-recovery contract.
This is an engineering assessment, not a full linguistic comparison.

## Included Hardening

- Fail-closed canonical source import and immutable, checksum-bound run inputs.
- Durable spend reservations and submission intent; ambiguous paid submissions
  require reconciliation instead of automatic replay.
- Bounded completion and recovery with preserved failure evidence.
- Catalog/sidecar validation and preserved publication identity and receipts.
- Exact-verse short-form validation, final encoded-audio semantic checks, and
  rejection of trailing speech from the next verse.
- Stricter per-ayah Urdu alignment checks and guarded, budget-limited content
  repair only when omission is proven.
- Consolidated dependencies, portable Urdu source-root selection, and explicit
  opt-in for tests requiring private production assets.

## Validation and Remaining Work

The consolidated clean-checkout suite was run with socket connections disabled:
**206 passed, 24 production-asset checks skipped**. The four standalone English
audio tests no longer depend on loading a frozen book in class setup. Production
checks are individually marked, retain their original assertions, and can be
enabled with `--production-assets`. The full asset-dependent suite is not
claimed as revalidated by this clean-checkout test.

The stricter historical Urdu alignment audit found 141/343 passing records and
202 requiring review. A review flag alone is not evidence of omitted audio. The
previously repaired unit 173 has validated mapped-word coverage of 0.982786;
preserve its repair and historical artifacts. The code update does not mutate,
regenerate, or republish existing media.

Provider model availability, account authorization, upload limits, and platform
processing remain external operational dependencies. No deployment, daily
scheduler installation, media upload, repository visibility change, or purchasing
action is part of this release.

## Local Artifacts and Provenance

Generated `output/` and `data/work/` trees, secrets, browser sessions, media, and
publication receipts stay outside Git. Release identifiers and pinned hashes
remain distinct from software version `2.0.0` and Git tag `v2`. Never update
frozen release fingerprints merely to make a resumed run accept different code.

Tanzil source attribution and third-party evidence notices remain intact.
Distribution rights for generated assets are not granted by this software tag.
