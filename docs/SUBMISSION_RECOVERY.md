# Recovering an uncertain batch submission

A `*.submission.json` receipt in state `submitting` without a batch ID means the
provider may already have accepted the paid request. It is not permission to retry.
Never delete the receipt, change its request hash, or use a new run ID to evade it.

## Freeze and collect evidence

1. Stop the affected worker and confirm that no other worker owns this run. Do not
   stop unrelated media or posting jobs. Preserve the original receipt, job files,
   frozen shard plan, request payload, timestamps, and logs in a recovery directory.
2. Reconstruct the exact ordered request list using the same frozen inputs and code.
   Its SHA-256 must match the submission receipt, using `json.dumps(requests,
   sort_keys=True, ensure_ascii=False).encode("utf-8")`. A mismatch is an integrity
   blocker, not a reason to replace the hash.
3. Use the provider console or read-only retrieval to identify the candidate batch
   in the correct account/project. Preserve the provider's response and request or
   support evidence locally without credentials. Confirm the batch ID, request
   count, custom IDs, model, and original payload association. Matching time/count
   alone is insufficient; custom IDs may recur across runs. The local request hash
   is not necessarily available from the provider.
4. If the association cannot be proven, keep the receipt blocked and contact provider
   support. An empty search result, missing local job, or timeout does not prove that
   no batch was accepted. Never automatically replay an uncertain paid submission.

## Adopt a proven existing batch

After an operator has reviewed the evidence, call `reconcile_batch_submission`
from `quran_translate.production_clients` with the same client, exact request list,
original job path, verified batch ID, and a local evidence file documenting the
payload association and reviewer. This function only retrieves the existing batch;
it never submits. It validates the intent hash and returned ID, stores an immutable
copy of the original receipt and evidence, then atomically records the known ID.
An existing different ID is rejected. Repeating adoption of the same ID is safe.

Resume the original stage normally. `submit_batch_once` will reuse the accepted ID;
the normal stage validators still check cached outputs and import remaining results.
Check that no new provider submission was made and preserve all recovery evidence.

The evidence JSON must contain `request_hash`, `batch_id`, `reviewer`,
`payload_association`, and `provider_evidence` (nonempty explanatory strings for the
last three). This is an operator attestation, not an automatic proof of payload
identity. Do not invent evidence just to satisfy validation. The provider response
must independently confirm the batch ID. Keep the referenced original evidence.

## Provider confirms nothing was accepted

Keep the original intent and written provider confirmation. There is intentionally
no automatic unblock/replay command. Escalate for a separately reviewed recovery
change with explicit spend authorization; do not erase ambiguity with a retry.

## Audio preparation

`audio-prepare --force` is deprecated and emits a warning. Identical inputs are
validated and reused; changed text, voice, or settings require a new `audio_run_id`.
It never overwrites completed audio. `audio-release --force` is a separate option.
