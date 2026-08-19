# Urdu Critic Benchmark Amendment v2

The first v1 call, to Gemini 3.7 Flash, was treated as a harness pilot. The model
identified all eight seeded defects with significant grounded findings and produced
no findings on the three clean controls. Five explicit Urdu-ledger violations were
classified as `ledger_error`, but v1 accepted only semantically adjacent labels such
as `sense_error` or `altered_scope`. The scorer therefore reported 3/8 recall even
though the finding text precisely named all eight defects.

Version 2 changes no passage, Urdu text, expected disposition, threshold, required
case, severity, prompt, schema, or clean control. It adds `ledger_error` as an
accepted type only for the six cases governed by an explicit Urdu ledger entry:
conjunction, *rafath*, *mubasharah*, *fajr*, *kalalah*, and *zina*.

The v1 manifest, response, score, and USD 0.0241965 cost remain preserved. Gemini's
identical response may be deterministically revalidated and rescored under v2 with
its provenance recorded by the harness's `rescore` command and no second provider
call. Terra is first called only
after v2 is frozen. Both candidates are therefore compared on the same v2 scoring
rule, while the outcome-informed amendment remains visible.
