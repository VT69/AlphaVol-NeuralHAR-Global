"""Build the development datasets (Step 2/5) into data/processed_v2/ with a manifest.

Usage:
  python scripts/build_dataset.py                                  # all assets, all groups
  python scripts/build_dataset.py --pending macro.credit_spread sentiment implied_vol.india_vix

Inputs that are being fetched locally (DECISIONS.md D7) can be listed under --pending: they are
omitted and recorded in the manifest. Anything else missing raises (no silent defaults).
Development data only; raw inputs are truncated at dev_end (D14).
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from alphavol import io as aio  # noqa: E402
from alphavol.config import ASSETS, load_config  # noqa: E402
from alphavol.features import GROUP_FEATURES, build_dataset, load_raw_inputs  # noqa: E402
from alphavol.logging_utils import configure, get_logger  # noqa: E402

log = get_logger("build_dataset")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--assets", nargs="+", default=list(ASSETS), choices=list(ASSETS))
    ap.add_argument("--pending", nargs="*", default=[], help="groups or group.feature allowed to be missing")
    args = ap.parse_args(argv)
    configure()
    cfg = load_config()
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    manifest: dict = {"git_commit": commit, "dev_end": str(cfg.dev_end.date()), "pending_requested": args.pending,
                      "assets": {}}
    for asset in args.assets:
        raw = load_raw_inputs(asset, cfg)
        res = build_dataset(asset, raw, cfg, groups=list(GROUP_FEATURES[asset]), pending=set(args.pending))
        path = cfg.processed_dir / f"{asset}_dataset.parquet"
        digest = aio.write_frame(res.data, path)
        manifest["assets"][asset] = {
            "file": str(path.relative_to(ROOT)), "sha256": digest, "rows": len(res.data),
            "first_date": str(res.data.index.min().date()), "last_date": str(res.data.index.max().date()),
            "last_target_date": str(res.data["target_date"].max().date()),
            "features": res.features, "pending_omitted": res.pending, "notes": res.notes,
            "provenance": raw.provenance,
        }
        log.info("%s: %d rows, groups %s, pending %s", asset, len(res.data), list(res.features), res.pending)
    aio.write_json(manifest, cfg.processed_dir / "manifest.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
