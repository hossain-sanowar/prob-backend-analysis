# ProB Backend Analysis

This repository contains the actual analysis notebooks used for the paper
"[Analysing ProB's Constraint Solving Backends: What do they know? Do they know things? Let's find out!](https://doi.org/10.1007/978-3-030-48077-6_8)"
submitted to [ABZ 2020](https://abz2020.uni-ulm.de/).

## Repo Outline

In this repository, you will find Python Notebooks as well as the CSV files
containing the data used for the analysis.

### Notebooks

On the top level, you will find the respective Python Notebooks in which we
carried out our evaluations.

* [`prob-f275-analysis.ipynb`](prob-f275-analysis.ipynb):
  Contains the analysis of the F275 feature set from previous work over whether
  ProB can find an answer to a given predicate or returns unknown
  (i.e. timeouts).
* [`prob-f109-analysis.ipynb`](prob-f109-analysis.ipynb):
  Contains the analysis of the F109 feature set over whether
  ProB can find an answer to a given predicate or returns unknown
  (i.e. timeouts).
* [`prob-f109-lto-analysis.ipynb`](prob-f109-lto-analysis.ipynb):
  Contains the analysis of the F109 feature set on a higher timeout (25 sec) for ProB's default backend.
* [`kodkod-f109-lto-analysis.ipynb`](kodkod-f109-lto-analysis.ipynb):
  Contains the analysis of the F109 feature set on a higher timeout (25 sec) for the Kodkod backend.
* [`z3-f109-lto-analysis.ipynb`](z3-f109-lto-analysis.ipynb):
  Contains the analysis of the F109 feature set on a higher timeout (25 sec) for the Z3 backend.
* The [`results`](results) directory contains the aggregated association rules
  collected over each backend's corresponding random forest.
  These are the 250,000 shortest rules found in the forests,
  for which support and confidence were calculated.
  Due to removing duplicates, the result files do not actually
  contain 250,000 rules each.

### Data

In the `data` directory, you will find *.zip files containing
data sets for each backend: `prob`, `kodkod`, and `z3`.

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

## Paper

The final authenticated version is freely available online at [Springer Link](https://link.springer.com/chapter/10.1007%2F978-3-030-48077-6_8).
If you plan on referencing this work, please use the following BibTex entry:

```bibtex
@inproceedings{dunkelau2020analysing,
  author={Dunkelau, Jannik and Schmidt, Joshua and Leuschel, Michael},
  title={Analysing {ProB}'s Constraint Solving Backends: What Do They Know? Do They Know Things? Let's Find Out!},
  booktitle={Rigorous State-Based Methods},
  series={LNCS},
  volume={12071},
  pages={107--123},
  year=2020,
  month=may,
  publisher={Springer},
  doi={10.1007/978-3-030-48077-6\_8}
}
```
