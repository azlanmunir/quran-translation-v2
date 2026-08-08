# Evidence Data

## Quranic Arabic Corpus Morphology

`qac-morphology.txt` is a pinned snapshot downloaded on 2026-08-03 from
[`mustafa0x/quran-morphology`](https://github.com/mustafa0x/quran-morphology), a
machine-readable representation of Quranic Arabic Corpus morphology distributed
under the GNU General Public License.

SHA256:

```text
742bfac59941b2cb09736d5b7aae694af50792261fb8450cbf6afafcc340645f
```

The production packets explicitly describe this morphology as scholarly annotation,
not raw fact. It constrains form but does not determine contextual meaning.

## Sense Ledger

`sense-ledger-v2.4.json` is the structured source of truth for contextual lexical
decisions. Every referenced evidence ID must resolve to an `evidence_sources` record;
the production preflight rejects dangling IDs.

## Refrain Policy

`refrain-policy-v1.json` stores only adjudicated English keyed by the SHA256 of
normalized Arabic. References are always rediscovered mechanically from the pinned
Tanzil source rather than maintained by hand.
