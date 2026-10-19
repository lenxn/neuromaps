# NeuroMaps: Disclaimer

This file is the release disclaimer required by the HEREDITARY misuse self-assessment (D9.1, Ethics Requirement 1). 
NeuroMaps is a visualisation tool: it can be used to illustrate regions of activity in fMRI images. 
The release does not contain personal data but the images to visualize must be provided by the user. 

> **Research use only. Not a medical device.**
> Developed in the HEREDITARY project as a research prototype. Not assessed or certified under the EU Medical Device Regulation (EU) 2017/745. Must not be used to diagnose, treat, monitor or decide anything about individual patients.


## 1. Overview

| Field | Value |
|---|---|
| Name | NeuroMaps |
| Version | v.1.2 |
| Type | Scripts for generating the described visualization. |
| Owner partner and contact | TU Graz (IVC); Stefan Lengauer, s.lengauer@tugraz.at |
| License | MIT (see LICENCE file) |
| Repository and DOI | https://github.com/lenxn/neuromaps — paper: https://doi.org/10.2312/evs.20261005 |
| Release date | Repository released with June 8, 2026 (corresponding to the presentation of the associated paper at EuroVIS) |
| Related task or deliverable | WP5, D5.6, T5.3 |
| How to cite | Lengauer, S., Kantz, B., Waldert, P., Tussardi, G., Kohn, N., Schreck, T., "NeuroMaps — A Compact Fingerprint for Analyzing Brain Activity," <i>EuroVis 2026</i>, 2026. [doi:10.2312/evs.20261005](https://doi.org/10.2312/evs.20261005) |

## 2. Intended use

**Primary intended use.** Research into efficient visualization of fMRI images.

**Intended users.** Neuroimaging and visualization researchers, data scientists.

**Out-of-scope uses.** This release must not be used for
- clinical diagnosis, prognosis or treatment of individual patients;
- decisions about individuals, such as insurance, employment or access to care;
- re-identification of individuals or linkage with other data sources;


## 3. Data provenance


This repository and the released files contain no personal data, pseudonymized records, identifiers or pseudonymization keys.



## 4. Method
- **Algorithm or architecture.** NeuroMaps is a brain-activity visualization approach that converts functional MRI (fMRI) activity maps into compact visual fingerprints. It takes a brain atlas and an activity/loading map as input and computes a saliency map, anatomical label contours, and projection vectors to summarize the spatial distribution of activity. The projection uses variance-based analysis and eigenvectors to represent activity patterns in a lower-dimensional space. The approach is a computational visualization method rather than a trained predictive machine-learning model. 

- **Training setting.** Not applicable. NeuroMaps does not describe a centralized or federated model-training procedure; it processes supplied brain-activity maps and an atlas to generate visualizations.

- **Privacy techniques.** Not applicable. NeuroMaps is a visualization tool, not a federated learning system.

## 5. Performance
- **Evaluation data.** The respective paper presents three illustrative examples of brain-activity z-score maps derived from a linked independent component analysis (LICA) of gut microbiota and brain activity. These examples were selected to demonstrate different visualization challenges. The repository also provides a manually simplified version of the Harvard–Oxford cortical and subcortical structural atlas for demonstration. 

- **Metrics and results.** The evaluation is primarily qualitative and visualization-focused. We observed that the resulting projections often align strongly with the sagittal, coronal, or horizontal planes, which we attribute to the mirror-symmetrical nature of many activity patterns. The method appears particularly useful when activity is concentrated in a few clearly separable regions. No conventional predictive-performance metrics, such as accuracy, sensitivity, specificity, or 95% confidence intervals have been used.

- **Results by subgroup.** No quantitative performance comparisons by age, sex, clinical subgroup, acquisition site, scanner, or demographic group are conducted. The three examples demonstrate different spatial visualization challenges rather than subgroup-specific model performance.

## 6. Known limitations and risks
- **Generalizability.** The evaluation is based on a small number of illustrative examples, so the robustness of NeuroMaps across larger and more diverse fMRI datasets remains uncertain. Generalizability to different populations, clinical conditions, acquisition protocols, scanners, atlases, and preprocessing pipelines has not been established through reported quantitative validation.

- **Known biases and confounders.** The method relies on the supplied brain atlas and the spatial characteristics of the activity map. Atlas resolution, anatomical parcellation choices, preprocessing, noise, and the distribution of activity across brain regions may influence the resulting visualization. Extensive overlap between regions in projection space can make the representation less effective. The influence of demographic and clinical confounders has not been systematically quantified.

- **Failure modes.** The principal limitation is severe overlap between brain regions in the projection space, which reduces the clarity and interpretability of the visualization. 

- **Privacy risks.** Because NeuroMaps is not a trained predictive model, membership-inference or model-inversion evaluations are not directly applicable in the usual model-training sense. However, the tool processes brain-activity data, which can be sensitive health-related information. Sharing generated visualizations or input data may still pose confidentiality or re-identification risks, depending on the data and accompanying metadata. 


## 7. Responsible use and reporting

- **Terms of use.** Public read-only demo; no user data upload path. Licensed under MIT [LICENSE](LICENSE).
- **Pre-release actions.** This disclaimer is linked from the repository README since 2026-10-19.
- **Contact for misuse or vulnerability reports.** s.lengauer@tugraz.at

## 8. Acknowledgement

Funded by the European Union under grant agreement No 101137074 (HEREDITARY). Views and opinions expressed are however those of the author(s) only and do not necessarily reflect those of the European Union or the European Health and Digital Executive Agency (HaDEA). Neither the European Union nor the granting authority can be held responsible for them.
