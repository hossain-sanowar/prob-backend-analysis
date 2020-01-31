"""
Script that gathers the results of the jobarray cluster scripts into a single
file, sorted by descending support score.

Script takes two arguments:

1. Path to result files of job array
2. Name of target file in which all data shall be gathered
"""
import re
import sys

from pathlib import Path


def extract_support(file):
    """
    Extracts the support score from the given association rule file.
    """
    with open(file, 'r') as f:
        for line in f.readlines():
            if line.startswith("Support:"):
                nums = [int(s) for s in re.findall(r'\b\d+\b', line)] # All numbers from line as list
                return nums[0]
        print("DID NOT FIND SUPPORT VALUE FOR", str(file))



if __name__ == "__main__":
    jobarray_dir = Path(sys.argv[1])
    target_file = sys.argv[2]

    # Gather all job dirs
    job_dirs = [d for d in jobarray_dir.iterdir() if d.is_dir()]

    # Gather all files per job dir
    rules = []
    for job in job_dirs:
        rules += [f for f in job.joinpath('job-parts/part').iterdir() if f.is_file()]

    # Collect support score
    print("Collect support scores")
    supp_rules = [(rule_file, extract_support(rule_file)) for rule_file in rules]
    sorted_rules = sorted(supp_rules, key=lambda r: r[1], reverse=True) # Sort by support

    # Dump into target file
    with open(target_file, 'w+') as dump:
        for (rule, support) in sorted_rules:
            if support <= 0: continue # Skipping unsupported rules.
            with open(rule, 'r+') as f:
                dump.write(f.read())
