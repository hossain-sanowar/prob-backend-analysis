"""
As the calculation of support and confidence per rule is O(n log(n))
and hence runs for a long while for 800,000 rules,
this file is meant to run it on a separate computer
(such as a HPC cluster node)
instead of a Jupyter Notebook.
"""

from f109_info import *
from intrees import *
from printing import *

import pandas as pd
import numpy as np
import pickle
import sys
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance

def pretty_print_rule(rule):
    cond, target = rule
    for (_, fid, thresh, leq) in cond:
        comp_sign = "<=" if leq else ">"
        print("# %3d: %s %s %.3f" % (fid, f109_name(fid), comp_sign, thresh))
    print("=> %d" % target)


def pretty_print_assoc_rule(rule, importances, target_file):
    (cond, target) = rule
    for c in sorted(cond, key=lambda x: 1/importances[x[0]]):
        fid, leq = c
        target_file.write(f109_name(fid))
        target_file.write(" (low)" if leq else " (**high**)")
        target_file.write(", importance: %.2f\n" % importances[fid])
    target_file.write("=> %d\n" % target)



def run_analysis(csv_file_path, target_dir='./'):
    data = pd.read_csv(csv_file_path)
    n_features = 109

    X = data[data.columns[0:n_features]]
    Y = data["Label0"]

    print("Training forest")
    forest = RandomForestClassifier(
        n_estimators=50, # 50 Trees.
        criterion="gini", # Using Gini index instead of "entropy"
        n_jobs=6, # Number of CPUs to use.
        bootstrap=False,
        max_features=0.7,
        random_state=123,
        class_weight="balanced")

    forest.fit(X, Y)
    print_classifier_stats(forest, X, Y)

    print("Calculating Gini importances")
    importances = forest.feature_importances_
    std = np.std([tree.feature_importances_ for tree in forest.estimators_],
                 axis=0)
    indices = np.argsort(importances)[::-1]

    print("Calculating permutation importances")
    # perm_importances = permutation_importance(forest, X, Y, scoring='balanced_accuracy', n_repeats=5, n_jobs=1, random_state=1234)
    # perm_indices = perm_importances.importances_mean.argsort()[::-1]

    print("Extracting rules")
    extracted_rules = extract_rules(forest)

    rule_count = 0
    for tree in forest.estimators_:
        rule_set = extracted_rules[tree]
        rule_count += len(rule_set)
    print("Collected %d rules" % rule_count)

    rule_list = []
    for tree in extracted_rules:
        rule_list += [(c,o) for (c,o) in extracted_rules[tree]]
    sorted_rule_list = sorted(rule_list, key=lambda r: len(r[0])) # Sorted rules by length

    md.write("# Listing of rules found by association rule analysis\n")
    md.write("\n")
    md.write("This list is sorted by descending support and confidence values.\n")
    md.write("\n")
    md.flush()

    print("Calculate association rule analysis")
    annotated_rules = analyse_rule_set(sorted_rule_list, max_depth=None)

    print("Save data to file")
    dat = open(target_dir+'/assoc_rules.dat', 'wb')
    pickle.dump(annotated_rules, dat)
    dat.close()

    md = open(target_dir+'/assoc_rule_overview.md', 'w+')

    seen = set([])
    for (cond, out, supp, conf) in sorted(annotated_rules, key=lambda r: (1/(r[2]+1), 1/(r[3]+1))):
        if not frozenset(cond) in seen:
            seen.add(frozenset(cond))
            pretty_print_assoc_rule((cond, out), permutation_importance.importances_mean, md)
            md.write('Support: %.2f%%, Confidence: %.2f\n\n' % (supp*100, conf))
    md.close()

if __name__ == "__main__":
    source = sys.argv[1]
    tar = sys.argv[2]
    Path(tar).mkdir(parents=True, exist_ok=True)
    print("Running for %s, data output to %s" % (source, tar))
    run_analysis(source, tar)
