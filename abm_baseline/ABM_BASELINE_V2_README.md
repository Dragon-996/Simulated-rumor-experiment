# Corrected ABM baseline (V2)

This directory contains a corrected, independently runnable version of the
Rumor Belief-Expression ABM (RBE-ABM) baseline. The original V1 files and `abm_results/` are not
modified. All V2 filenames and outputs carry the `_v2` suffix or use
`abm_results_v2/`.

## Scope and terminology

V2 is a simple pairwise-network rumor baseline, not a full substitute for the
LLM model's natural-language debunking or semantic polarization experiments.
It is intended to test whether the aggregate rumor-diffusion result requires
LLM-generated decisions.

The network stored in `net_hyper_100_seed46.json` is displayed as
**Community** because the file contains an ordinary adjacency list (a
pairwise projection), not explicit hyperedges or a higher-order update rule.
The filename is retained only to preserve compatibility with the existing
data file.

## Node states

The dynamic state is the Cartesian product of private belief and public focal
participation:

| Belief | Participation | State | Focal post |
|---|---|---|---|
| believer | active | active spreader | rumor |
| believer | silent | silent believer | none |
| non-believer | active | active skeptic | skeptical |
| non-believer | silent | passive skeptic | none |

Silence therefore creates no neutral post and does not enter a neighbor's
focal-event feed.

## Corrected transition mechanism

For agent `i`, let `R_i(t)` and `D_i(t)` be the observed rumor and skeptical
posts at round `t`. The per-contact rumor probability is `tau_i`; the
per-contact skeptical correction probability is `delta`. The two pressures
are calculated separately:

```text
rumor_pressure_i = 1 - (1 - tau_i) ^ R_i
skeptical_pressure_i = 1 - (1 - delta) ^ D_i
```

Prior belief is first retained with probability `rho`, then rumor contacts can
produce adoption, and skeptical contacts can only reduce the resulting belief
probability:

```text
belief_before_skepticism
  = 1 - (1 - rho * previous_belief) * (1 - rumor_pressure_i)

P(belief_i(t+1))
  = belief_before_skepticism * (1 - skeptical_pressure_i)
```

The default `rho=1` removes spontaneous belief decay when no counter-message
is observed. Belief and participation are sampled once each; exposure is not
multiplied a second time, avoiding the former approximate exposure-squared
barrier.

Believers publish rumor posts with a sharing probability. Non-believers can
publish skeptical posts only after rumor exposure. Previous public activity
may persist through a separate participation-persistence probability.

## ABM variants

- **ABM-0** uses homogeneous effective traits and shared transition
  probabilities.
- **ABM-H** uses the same equations and global parameters, with Big Five
  composites adjusting rumor-contact and believer-sharing probabilities.

No parameter is fitted separately for a topology or model variant.

## Network and seed controls

The four existing node sets, attributes, and adjacency lists are reused
directly. Network generation is not repeated.

Two seed analyses are run:

- `structural`: the five highest-degree nodes in each network are selected.
  This matches an influential-seed intervention but seed identities differ by
  topology.
- `paired_random`: each Monte Carlo replicate uses the same five agent IDs in
  all four networks and both ABM variants. This controls identity and
  personality while allowing the selected nodes' structural positions to
  differ across networks.

Because the supplied networks still differ in edge count and ER contains an
isolate, topology results should be described as comparisons among these four
fixed network instances, not as density-matched causal effects of topology.

## Files and commands

```text
abm_baseline_v2.py             corrected simulation core
abm_parameters_v2.json         shared parameters and fixed network mapping
run_abm_experiments_v2.py      Monte Carlo runner and 95% t intervals
plot_abm_comparison_v2.py      penetration/entropy and peak figures
test_abm_baseline_v2.py        mechanism regression tests
```

Run from this directory:

```powershell
python -m unittest test_abm_baseline_v2.py
python run_abm_experiments_v2.py
python plot_abm_comparison_v2.py
```

The runner saves raw run JSON files, a timestamped summary, and
`abm_results_v2/abm_v2_latest_summary.json`. The plotter exports editable SVG
and PDF, 600-dpi TIFF and PNG, and CSV source data.

## Interpretation boundary

These shared parameter values were selected by simple preliminary fitting and
then frozen across all networks, seed conditions, and model variants. They are
not human behavioral parameter estimates, and this baseline is not presented
as a calibrated forecasting model. The included seed-strategy check is a
limited sensitivity analysis rather than a global calibration exercise.
The baseline supports a claim about comparative aggregate diffusion only; it
does not validate the semantic content, reasoning, debunking quality, or
language-based polarization produced by LLM agents.
