# project_001: Chinese drama → Hindi remake

This is a personal self-improvement learning project. Outputs are private and are not distributed.

| Item | Value |
|---|---|
| Source | `input/source.mp4`, hardlinked from the legacy workspace `../../input/source.mp4`. sha256 `69eb4aed…7862f`. |
| Duration / format | 6669.58 s · 720×1280 · 30 fps · H.264 + AAC |
| Languages | zh → hi (IN) |
| Legacy analysis | Imported with `--migrate-legacy` as cached artifacts. See `migration/manifest_v0.json`. |

Commands:

```
./run.sh --project projects/project_001/config.yaml --validate-only
./run.sh --project projects/project_001/config.yaml --migrate-legacy .
./run.sh --project projects/project_001/config.yaml --status
./run.sh --project projects/project_001/config.yaml --plan --mode analyze
```
