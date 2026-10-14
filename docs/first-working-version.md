# First working version

Local baseline accepted by the user before legacy cleanup:

- Git tag: `first-working-v1` (annotated)
- Commit: `7457770cc4b1aa09872ca0e33af43a823dbcb6fc`
- Baseline automated verification: `619 passed, 1 skipped`.
- This is a user-accepted working baseline, not a claim of exhaustive native validation.

## Preserved behavior

- Direct minimap texture change plus route extent, with five-second confirmation.
- Hidden minimap cannot establish ARRIVED/STUCK.
- Local popup/voice, resource warnings, repeat-alert settings.
- In-app minimap recorder with explicit start/stop and manifest finalization.
- Ukrainian/English interface and settings save feedback.

## Scope of the snapshot

The tag preserves tracked source, tests and assets. It does **not** back up local
calibration, settings, raw recordings or installed dependencies. Those remain at
their existing local paths and must not be removed during source cleanup.

To inspect without changing the working tree:

```bash
git show first-working-v1
git diff first-working-v1 -- bdo_assistant tests
```

Do not reset the current working tree to this tag merely to inspect it. Any
rollback of later changes should be a separate explicitly approved action.
