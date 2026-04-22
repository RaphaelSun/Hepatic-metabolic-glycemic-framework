# Hepatic-metabolic-glycemic-framework
Analysis code and figure-generation scripts for a severity-based hepatic-metabolic-glycemic framework linking MASLD phenotypes to proteinuria and cardio-kidney risk in a hospital health-check cohort.


# Hepatic-Metabolic-Glycemic Framework

Analysis code and figure-generation scripts for a severity-based hepatic-metabolic-glycemic framework linking hepatic steatosis, metabolic burden, glycemic status, and multi-organ risk in a hospital health-check cohort.

## Overview

This repository contains the core analysis scripts and figure-generation scripts used to evaluate a severity-based hepatic-metabolic-glycemic phenotyping framework in a hospital health-check cohort. The analytical workflow focuses on:

- construction of the full analytic cohort and echocardiographic subset
- overall phenotype-to-outcome analyses
- glycemic-stratified proteinuria analyses
- insulin-resistance characterization
- sensitivity analyses
- discrimination analyses
- reproduction of the main manuscript figures

The framework combines:

- hepatic steatosis severity: no steatosis, mild steatosis, moderate/severe steatosis
- non-glycemic metabolic burden: low, intermediate, high
- glycemic status: normoglycemia, intermediate hyperglycemia, diabetes-level hyperglycemia

Primary and supportive outcomes include:

- proteinuria
- cardio-kidney co-abnormality
- diastolic dysfunction
- insulin-resistance proxy characterization using TyG and METS-IR

## Data availability

Participant-level source data are **not included** in this repository.

The analysis scripts expect a frozen hospital cohort file and a flow-summary table located at:

- `outputs/cohorts/hospital_main_cohort_frozen.csv.gz`
- `outputs/tables/tableS1_variable_availability_and_flow.csv`

These files are not distributed publicly because they are derived from hospital health-check data. Users who wish to rerun the full analysis will need access to the corresponding source dataset.

## Repository contents

This repository includes:

### Core analysis scripts

- `new/scripts/rebuild_lib.py`
- `new/scripts/00_build_rebuild_cohorts.py`
- `new/scripts/01_run_overall_outcomes.py`
- `new/scripts/02_run_glycemic_interface.py`
- `new/scripts/03_run_sensitivity.py`
- `new/scripts/04_build_results_snapshot.py`
- `new/scripts/05_run_ir_characterization.py`
- `new/scripts/06_run_roc_comparison.py`

### Figure-generation scripts


- `new/scripts/fig2_ir_support.py`
- `new/scripts/fig3_glycemic_interface.py`
- `new/scripts/fig4_overall_proteinuria.py`
- `new/scripts/fig5_cardio_kidney_composition.py`

## Figure mapping

The main manuscript figures are generated as follows:

- **Figure 1**: study flow  

- **Figure 2**: insulin-resistance support layer  
  - analysis input: `05_run_ir_characterization.py`
  - plotting script: `fig2_ir_support.py`

- **Figure 3**: MASLD-glycemic interface  
  - analysis input: `02_run_glycemic_interface.py`
  - plotting script: `fig3_glycemic_interface.py`

- **Figure 4**: overall proteinuria backbone  
  - analysis input: `01_run_overall_outcomes.py`
  - plotting script: `fig4_overall_proteinuria.py`

- **Figure 5**: cardio-kidney composition  
  - analysis input: `01_run_overall_outcomes.py`
  - plotting script: `fig5_cardio_kidney_composition.py`

- **Figure 6**: discrimination / ROC comparison  
  - analysis and figure generation: `06_run_roc_comparison.py`
