# Proposed Model 3 — SAM2 Memory Tree Re-acquisition

This version is self-contained inside Target_Locker. It does not use SAMURAI.

## Pipeline

```
RGB video
↓
Click target on frame 1
↓
SAM2.1 Tiny tracking
↓
Check mask quality
↓
Lock healthy?
├── Yes → keep tracking and store reliable target appearance
└── No  → start Memory Tree Search
             ↓
        generate several candidate target locations
             ↓
        create multiple branches
             ↓
        score each branch using:
          - appearance similarity
          - motion continuity
          - target-size consistency
          - cumulative path score
             ↓
        prune weak branches
             ↓
        keep top hypotheses across frames
             ↓
        stable strong branch confirmed?
             ├── No  → continue tree search
             └── Yes → re-prompt SAM2 at that candidate
                         ↓
                      continue tracking
```

## Memory Tree

`memory_tree.py` implements an independent SAM2Long-inspired beam-search memory tree.

It does not copy SAM2Long source code. It adapts the general idea of maintaining multiple hypotheses instead of immediately committing to one candidate.

The tree keeps up to five active branches. During target loss, each branch receives a cumulative score based on appearance, motion continuity and size consistency. A candidate must remain stable across multiple lost frames before it is allowed to reinitialize SAM2.

## No-GT diagnostics

The app reports only diagnostics:

- loss events
- memory-tree search frames
- confirmed re-acquisitions
- mean best tree score
- mean SAM confidence on locked frames
- normalized center jump
- tracking FPS

These are not tracking-accuracy metrics because no ground truth is used.

## Files

- `App Interface.ipynb` — Colab setup
- `app_interface.py` — upload / click / output UI
- `model3_tracker.py` — SAM2 tracking + loss detection + tree re-lock
- `memory_tree.py` — multi-hypothesis memory-tree search
- `requirements.txt` — dependencies
