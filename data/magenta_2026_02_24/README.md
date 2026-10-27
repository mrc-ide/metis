# **\<\>\_realisations.parquets**

This folder contains stochastic simulation outputs from an individual-based malaria transmission and drug-resistance model (Magenta). The data are intended for training and validating statistical or machine-learning surrogate models that emulate the behaviour of the full IBM.

This description is for realisations.parquet, non_malarial_fever_realisations.parquet and non_malarial_fever_and_high_resistance_realisations.parquet. Each are outputs from the same model but with different structural model assumptions about presence of non-malarial fever (nmf) and whether resistance is higher than current data suggests. 

**NOTE:** To help looking for and thinking about the blip when resistance comes in, I have given you some of the simulation prior to resistance kicking in. `t` now starts at 10 years and resistance is switched on at 20 years (sorry forgot to adjust to it to \-10 start). So at `t`\>20, you will start to see patterns changing more. Also no strong guarantee that model is at perfect equilibrium at `t = 10` years, but likely will be just before 20 years. 

Each row corresponds to the model output at a given time point from a single stochastic realisation under a specific set of input parameters.

### **Structure of the data**

The dataset is a rectangular table with one row per `(parameter set, replicate, time)` combination. Not all parameter sets have the same number of replicates, as some simulations terminate early (for example, due to elimination or numerical instability).

### **Indexing and bookkeeping variables**

* **`rep`**  
  Integer identifier for the stochastic replicate. Replicates are independent realisations of the same underlying parameter set.  
* **`t`**  
  Time since the start of the observation window, in years. Outputs are recorded at regular intervals following model burn-in.

### **Input parameters**

These variables define the intervention, resistance landscape, and epidemiological setting used in the IBM simulation. They are treated as inputs when training the surrogate model.

* **`al`**  
  Proportion of total treatments represented by artemether–lumefantrine.  
* **`asaq`**  
  Proportion of total treatments represented by artesunate–amodiaquine.  
* **`dhappq`**  
  Proportion of total treatments represented by dihydroartemisinin–piperaquine.  
* **`art_res`**  
  Starting population frequency of artemisinin resistance.  
* **`ppq_res`**  
  Starting population frequency of piperaquine resistance.  
* **`aq_res`**  
  Starting population frequency of amodiaquine resistance.  
* **`lu_res`**  
  Starting population frequency of lumefantrine resistance.  
* **`eir`**  
  Daily entomological inoculation rate determining transmission intensity.  
* **`ft`**  
  Frequency of treatment representing treatment-seeking behaviour and drug pressure.

### **Model outputs**

These variables are the simulated epidemiological and evolutionary outcomes produced by the IBM and form the targets for surrogate model training.

* **`cases`**  
  Annual malaria case incidence per person without additional cases due to treatment failure driven by resistance.  
* **`prev`**  
  Malaria prevalence by PCR.  
* **`micro210`**  
  Malaria prevalence by microscopy diagnosis in children aged 2–10 years.  
* **`tf`**  
  Average treatment failure due to resistance.  
* **`lpf`**  
  Annual late parasitological failure incidence per person due to resistance.  
* **`lcf`**  
  Annual late clinical failure incidence per person due to resistance.  
* **`all_cases`**  
  Total annual malaria case incidence per person.  
* **`deaths`**  
  Annual malaria mortality incidence per person.  
* **`lcf_deaths`**  
  Additional annual malaria mortality incidence per person due to resistance-associated late clinical failure.  
* **`a_1`**  
  Population allele frequency for genetic marker 1 associated with resistance.  
* **`a_2`**  
  Population allele frequency for genetic marker 2 associated with resistance.  
* **`a_3`**  
  Population allele frequency for genetic marker 3 associated with resistance.  
* **`a_4`**  
  Population allele frequency for genetic marker 4 associated with resistance.  
* **`a_5`**  
  Population allele frequency for genetic marker 5 associated with resistance.  
* **`a_6`**  
  Population allele frequency for genetic marker 6 associated with resistance.

All outputs are stochastic and reflect both demographic noise, transmission variability, and evolutionary dynamics in the IBM.

### 

### 

### 

### **Replicates and missingness**

* The number of replicates (`rep`) may vary between parameter combinations.

* Some simulations terminate early if malaria is eliminated or the model becomes numerically unstable and these then crashed and missing, hence unequal reps per parameter combination

Surrogate models should therefore be robust to:

* Unequal numbers of replicates

* Correlation of outputs within replicates over time

