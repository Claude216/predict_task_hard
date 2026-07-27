"""
One meta-feature file PER DATASET (127 files) under results/meta_features/.

Each file {dataset}.csv is long-format `kind,name,value` holding:

  x_variation  <feature>            standardized [0,1] variation of each
                                    feature column (targets excluded)
  correlation  <feat1>|<feat2>      correlation of every pair of features
                                    ([0,1]: |Pearson| num-num, Cramer's V
                                    sym-sym, correlation ratio num-sym)
  y_variation  <target>             standardized [0,1] variation of each
                                    optimization-target column
  performance  rmse                 EZR performance score, DEFAULT settings
  stability    agreement            EZR stability score, DEFAULT settings
  stability    jaccard_mean         mean pairwise Jaccard of the 20 trees'
                                    feature sets (same trees as `agreement`)

Variation / correlation values are taken from analysis/meta_features_detail.csv
(produced by analysis/compute_meta_features.py; definitions documented there).

Performance and stability replicate 9.py's "ezr" treatment EXACTLY, except
that `the` keeps ezr.py's DEFAULT settings (acq=near, Any=4, Budget=30,
Check=5, leaf=3, Impurity=gini) instead of 9.py's refined overrides
(Budget=50, Check=10):
  - performance rmse: 20 train/holdout splits (seeds 0,10,...,190); per split
    err = win(best reachable in holdout+labels) - win(best of tree-guided
    Check picks or labels); rmse = sqrt(mean err^2).       [9.py Part 1]
  - stability agreement: fixed seed-42 half split; 20 trees (seeds 0..19)
    from likely() labels; per test row (<=100), the sd of the 20 win-scores;
    agreement = % rows with sd < 0.35 * sd(win over whole data). [9.py Part 2]
  - jaccard_mean: on those same 20 trees, weighted Jaccard of split-feature
    sets (jaccard.py's features_used_in_tree / jaccard).

Run: /Users/claudeli/Tool/miniconda3/envs/caus26/bin/python analysis/per_dataset_meta.py
"""
import csv as _csv
import glob
import itertools
import os
import random
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

REPO = "/Users/claudeli/NCSU/Research/26sp_aise/stability_evaluation"
sys.path.insert(0, REPO)

from ezr import (Data, adds, clone, csv, disty, likely, shuffle,  # noqa: E402
                 the, Tree, treeLeaf)
from jaccard import features_used_in_tree, jaccard  # noqa: E402

OUT_DIR = os.path.join(REPO, "results/meta_features")
DETAIL = os.path.join(REPO, "analysis/meta_features_detail.csv")
REPEATS = 20


def ezr_default_scores(path):
    "9.py's 'ezr' performance and stability, under ezr.py default settings."
    assert (the.Budget, the.Check, the.acq, the.leaf, the.Impurity) == \
           (30, 5, "near", 3, "gini"), "the-defaults changed?"
    all_data = Data(csv(path))
    if not all_data.cols.y and all_data.cols.klass:
        all_data.cols.y = [all_data.cols.klass]

    ys      = [disty(all_data, row) for row in all_data.rows]
    b4      = adds(ys)
    win     = lambda v: int(100 * (1 - (v - b4.lo) / (b4.mu - b4.lo)))
    b4_wins = adds([win(k) for k in ys])

    # ---- performance (9.py Part 1, trt == "ezr") ----
    mse = 0
    for rand_seed in range(REPEATS):
        random.seed(rand_seed * 10)
        the.seed      = rand_seed * 10
        shuffled_rows = random.sample(all_data.rows, len(all_data.rows))
        half          = int(0.5 * len(all_data.rows))
        train         = clone(all_data, shuffled_rows[:half])
        holdout_rows  = shuffled_rows[half:]
        labels        = likely(train)
        tree          = Tree(clone(train, labels))
        top_rows = sorted(
            [(treeLeaf(tree, row).mu, row) for row in holdout_rows],
            key=lambda x: x[0])[:the.Check]
        trt_perf = max(
            win(min(disty(all_data, row) for _, row in top_rows)),
            win(min(disty(all_data, row) for row in labels)))
        ref_opt = win(min(disty(all_data, row) for row in holdout_rows + labels))
        mse += abs(ref_opt - trt_perf) ** 2
    rmse = (mse / REPEATS) ** 0.5

    # ---- stability (9.py Part 2, trt == "ezr") ----
    random.seed(42)
    all_data.rows = shuffle(all_data.rows)
    half       = len(all_data.rows) // 2
    train      = clone(all_data, all_data.rows[:half])
    test_rows  = all_data.rows[half:][:min(100, len(all_data.rows) - half)]
    consistent_Y = lambda row: disty(all_data, row)

    trees = []
    for rand_seed in range(REPEATS):
        the.seed = rand_seed
        random.seed(rand_seed)
        labels = likely(train)
        trees.append(Tree(clone(train, labels), Y=consistent_Y))

    agreement = 0
    for row in test_rows:
        win_scores = [win(treeLeaf(tree, row).mu) for tree in trees]
        if adds(win_scores).sd < 0.35 * b4_wins.sd:
            agreement += 1
    agreement = agreement * 100 // len(test_rows)

    feature_sets = [features_used_in_tree(all_data, tree) for tree in trees]
    js = [jaccard(feature_sets[i], feature_sets[j])
          for i, j in itertools.combinations(range(REPEATS), 2)]
    jaccard_mean = sum(js) / len(js)

    return rmse, agreement, jaccard_mean


def one_dataset(args):
    path, detail_rows = args
    dataset = os.path.basename(path)[:-4]
    rmse, agreement, jaccard_mean = ezr_default_scores(path)
    out_path = os.path.join(OUT_DIR, f"{dataset}.csv")
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        w = _csv.writer(fh)
        w.writerow(["kind", "name", "value"])
        w.writerows(detail_rows)  # x_variation, correlation, y_variation
        w.writerow(["performance", "rmse", f"{rmse:.2f}"])
        w.writerow(["stability", "agreement", agreement])
        w.writerow(["stability", "jaccard_mean", f"{jaccard_mean:.4f}"])
    return dataset, rmse, agreement, jaccard_mean


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    files = sorted(glob.glob(os.path.join(REPO, "data/optimize/*/*.csv")))
    assert len(files) == 127, f"expected 127 files, got {len(files)}"

    detail = {}  # dataset -> [(kind, name, value), ...] in original order
    with open(DETAIL, encoding="utf-8") as fh:
        for r in _csv.DictReader(fh):
            detail.setdefault(r["dataset"], []).append(
                (r["kind"], r["name"], r["value"]))
    assert set(detail) == {os.path.basename(f)[:-4] for f in files}

    # biggest files first so the pool stays balanced
    files = sorted(files, key=os.path.getsize, reverse=True)
    jobs = [(f, detail[os.path.basename(f)[:-4]]) for f in files]

    done = 0
    with ProcessPoolExecutor(max_workers=7) as ex:
        futs = [ex.submit(one_dataset, j) for j in jobs]
        for fut in as_completed(futs):
            dataset, rmse, agreement, jm = fut.result()
            done += 1
            print(f"[{done}/{len(files)}] {dataset}: rmse={rmse:.2f} "
                  f"agreement={agreement} jaccard={jm:.4f}", flush=True)
    print(f"\nwrote {done} files to {OUT_DIR}")


if __name__ == "__main__":
    main()
