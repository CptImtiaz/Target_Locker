# Proposed Model 3 — Rolling Target Memory Recovery

Model 3 uses one recovery strategy only.

## Pipeline

```
RGB video
↓
Select target on frame 1
↓
SAM2 tracking
↓
While lock is healthy:
    save target crop from the tracking box
    keep recent target appearances in memory
↓
Lock lost?
├── No  → keep tracking + update memory
└── Yes → search the whole frame using saved target memories
          ↓
       best matching target candidate
          ↓
       re-initialize SAM2
          ↓
       continue tracking
```

## Recovery idea

The model stores:
- the original target crop as an anchor
- recent target crops from healthy tracked frames
- multiple target scales

After lock loss, it compares the current frame against those stored target memories at several scales. Multiple stored memories vote for the same location, which helps reduce false re-locks.

## Evaluation

Every run reports:
- Lock retention %
- Lock-loss events
- Re-acquisition attempts
- Successful re-acquisitions
- Re-acquisition success %
- Mean recovery latency
- Mean SAM confidence
- Mean normalized center jump
- Tracking FPS

These are proxy metrics. Formal SOT evaluation should still use ground-truth Success/AUC, Precision and Normalized Precision.

## Files

- `App Interface.ipynb` — Colab setup and launcher
- `app_interface.py` — upload/select/track UI
- `model3_tracker.py` — SAM2 tracking, rolling memory bank, loss detection and re-lock
- `requirements.txt` — runtime dependencies
