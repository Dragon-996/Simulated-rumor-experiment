
# RBE-ABM V2 baseline

This directory contains the two baseline variants used for the controlled
comparison:

- `ABM-0`: homogeneous agents using population-mean trait values;
- `ABM-H`: the same transition rules with fixed Big Five-derived
  heterogeneity.

Both variants use the same four fixed networks, seed schedules, rounds, and
shared transition parameters. The parameters were selected by simple
preliminary fitting and then held fixed. They are not estimates of human
behavioral parameters.

Run tests and the complete experiment:

```powershell
python -m unittest test_abm_baseline_v2.py
python run_abm_experiments_v2.py --config abm_parameters_v2.json
```

For a quick local smoke test:

```powershell
python run_abm_experiments_v2.py `
  --config abm_parameters_v2.json `
  --runs 1 `
  --rounds 1 `
  --output-dir smoke_results
```

`abm_baseline.py` is the source data loader required by V2. It verifies fixed
agent attributes across networks and maps Big Five values as follows:

- belief susceptibility = `(Openness + (1 - Conscientiousness) + Neuroticism) / 3`;
- conformity = `Agreeableness`;
- expression = `(Extraversion + Neuroticism) / 2`.

The file named `net_hyper_100_seed46.json` is analyzed as a community-network
binary projection. No higher-order interaction is implemented or claimed.

Archived outputs are under `results/`; TIFF submission files are intentionally
excluded from Git.
