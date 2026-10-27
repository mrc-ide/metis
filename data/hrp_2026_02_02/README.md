# hrp_2026_02_02.parquet

This file contains stochastic simulation outputs from an individual-based malaria transmission model. The data are intended for training and validating statistical or machine-learning surrogate models that emulate the behaviour of the full IBM.

Each row corresponds to the model output at a given time point from a single stochastic realisation under a specific set of input parameters.

### Structure of the data

The dataset is a rectangular table with one row per (parameter set, replicate, time) combination. Not all parameter sets have the same number of replicates, as some simulations terminate early (for example, due to elimination).

### Indexing and bookkeeping variables

- **rep**  
Integer identifier for the stochastic replicate. Replicates are independent realisations of the same underlying parameter set.

- **t**  
Time since the start of the observation window, in years. Outputs are recorded at regular intervals following model burn-in.

### Input parameters

These variables define the intervention and epidemiological setting used in the IBM simulation. They are treated as inputs when training the surrogate model.

- **EIR**  
Entomological inoculation rate - determines mosquito biting which drives most of the pattern in Micro.2.10.

- **ft**  
Treatment-seeking or follow-up parameter controlling access to care.

- **microscopy.use**  
Proportion of cases diagnosed using microscopy rather than rapid diagnostic tests (RDTs).

- **rdt.nonadherence**  
Probability of non-adherence to RDT results (for example, treatment given despite a negative test).

- **fitness**  
Relative fitness of resistant parasites compared to sensitive parasites.

- **rdt.det**  
Indicator for whether RDT-based diagnosis is used (binary).

- **Micro.2.10**  
Mean microscopy prevalence in children aged 2–10 years during the final year of burn-in.  
Although this quantity is an output of the IBM, it is included here as an *effective input* for surrogate modelling. The underlying IBM is parameterised in terms of entomological inoculation rate (EIR), but the surrogate model is intended to be conditioned on observable prevalence rather than EIR directly.

### Model outputs

These variables are the simulated epidemiological outcomes produced by the IBM and form the targets for surrogate model training.

- **micro\_2\_10**  
Microscopy prevalence in children aged 2–10 years.

- **pcr**  
PCR-based parasite prevalence.

- **clinical\_05**  
Clinical malaria incidence in children under 5 years.

- **clinical**  
Clinical malaria incidence across all ages.

- **severe\_05**  
Severe malaria incidence in children under 5 years.

- **severe**  
Severe malaria incidence across all ages.

- **mortality\_05**  
Malaria-attributable mortality rate in children under 5 years.

- **mortality\_100**  
Malaria-attributable mortality rate across all ages.

All outputs are stochastic and reflect both demographic noise and transmission variability in the IBM.

### Replicates and missingness

- The number of replicates (rep) may vary between parameter combinations.

- Some simulations terminate early if malaria is eliminated or the model becomes numerically unstable and these then crashed and missing, hence unequal reps per parameter combination

Surrogate models should therefore be robust to:

- Unequal numbers of replicates

- Correlation of outputs within replicates over time
