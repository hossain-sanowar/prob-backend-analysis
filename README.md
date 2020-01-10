# ProB Backend Analysis

This repository contains the actual analysis notebooks used for the paper
"Analysing the Backends of ProB:
  What do they know? Do they know things? Let's find out!"
to be submitted at the ABZ 2020.

## Repo Outline

In this repository, you will find Python Notebooks as well as the CSV files
containing the data used for the analysis.

### Notebooks

On the top level, you will find the respective Python Notebooks in which we
carried out our evaluations.

* [`prob_f275-analysis.ipynb`](prob_f275-analysis.ipynb):
  Contains the analysis of the F275 feature set over whether
  ProB can find an answer to a given predicate or returns unknown
  (i.e. timeouts).

### Data

In the `data` directory, you will find *.zip files containing
data sets for each backend, `prob`, `kodkod`, and `z3`.

For each backend, six files exists:
each of the feature sets F17, F185, and F275
is present in two collections with different suffixes:

* suffix `_all`:
  containins a data sample for each predicate from the data source
* suffix `_unique`:
  contains the same data samples as the `_all` version,
  but sorted and all entries appear uniquely.
  As the unique filter does reduce the feature set by 2/3rds,
  it is apparent that the features are not discriminatory enough and lack
  certain metrics.

## Data Source

The data used for this experiments was taken from the
[Satisfiability Overview of Predicates for ProB Backends](https://github.com/hhu-stups/prob-examples-metadata/tree/master/b-predicates)
compiled in earlier work.
