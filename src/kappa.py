"""Inter-annotator agreement for the manual error coding sheet (results/audit/manual_coding_sheet.csv).
Reports Cohen's kappa between coder1 and coder2 on error_type and on invented_detail, plus
agreement between the automatic error_type and the adjudicated final_error_type."""
from __future__ import annotations

import argparse

import pandas as pd
from sklearn.metrics import cohen_kappa_score

from common import load_config, p


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--tag", default=""); args = ap.parse_args()
    cfg = load_config()
    d = pd.read_csv(p(cfg["paths"]["audit_dir"]) / f"manual_coding_sheet{args.tag}.csv", dtype=str, keep_default_na=False)
    both = d[(d.coder1_error_type != "") & (d.coder2_error_type != "")]
    if len(both):
        print(f"error_type kappa (coder1 vs coder2, n={len(both)}): {cohen_kappa_score(both.coder1_error_type, both.coder2_error_type):.3f}")
        c1, c2 = both["coder1_invented_detail(Y/N)"], both["coder2_invented_detail(Y/N)"]
        print(f"invented_detail kappa: {cohen_kappa_score(c1, c2):.3f}")
    fin = d[d.final_error_type != ""]
    if len(fin):
        print(f"auto vs adjudicated agreement (n={len(fin)}): {(fin.error_type == fin.final_error_type).mean():.3f}, "
              f"kappa {cohen_kappa_score(fin.error_type, fin.final_error_type):.3f}")
    if not len(both) and not len(fin):
        print("Coding sheet not filled in yet.")


if __name__ == "__main__":
    main()
