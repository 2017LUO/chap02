# Three-Seed Simulation Reference Data

This directory is a copied snapshot of the simulation training data from `Paper_Code`.
It is kept inside the reproduction project so later real-environment reproduction runs
can compare against the same seed set and directory layout.

## Layout

```text
results/reference_simulation/
├── reference_manifest.json
├── chap02/
│   ├── comparison/{patrol_parking,road_parking}/three_seed_runs/
│   └── ablation/{patrol_parking,road_parking}/three_seed_runs/
├── chap03/
│   ├── comparison/{patrol_parking,road_parking}/three_seed_runs/
│   └── ablation/{patrol_parking,road_parking}/three_seed_runs/
└── chap04/
    ├── comparison/{patrol_parking,road_parking}/three_seed_runs/
    └── ablation/{patrol_parking,road_parking}/three_seed_runs/
```

Each `three_seed_runs` suite keeps `seed_42`, `seed_43`, and `seed_44`.
`road_parking` maps to the reproduction environment `ugv_parking`; `patrol_parking`
maps to `ugv_patrol`.

## Training Entry

Use the unified reproduction entry from the project root:

```bash
conda run -n DT python scripts/train/run_reproduction_training.py --chapter chap03 --group comparison --scene road_parking --seeds 42 43 44 --episodes 4000
```

For a quick layout check without running training:

```bash
conda run -n DT python scripts/train/run_reproduction_training.py --chapter all --group comparison --scene all --dry-run
```

The generated reproduction outputs use:

```text
results/reproduction_training/chapXX/{comparison|ablation}/{road_parking|patrol_parking}/three_seed_runs/seed_*/_workspace/<method>/
```
