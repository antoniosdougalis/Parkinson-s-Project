# Multi-Head Attention-Transformer Architecture for EEG Feature classification in Parkinson's Disease

A Python-based Pre-Processing and Deep-Learning pipeline of scripts for the submitted manuscript 

Antonios. G. Dougalis. 2026, Interpretable Electrophysiological Features of Resting-State EEG Capture Cortical Network Dynamics in Parkinson’s Disease, 
arXiv;

The manuscript is available as a pre-print at arXiv at: https://arxiv.org/html/2604.01475v3

The files used to perform the Deep Learning Classification on the EEG extracted features are freely available for public use:

1. jpersMed_myReduced_Transformer.py
-Multi-Head Attention-Transformer Architecture for EEG Feature classification

2. projParkin_Subj_DL_Main_Github.py
-Main Script for the implementation of the Transformer Deep Learning classification on EEG Features using a Leave-One-Subject-Out (LOSO) procedure. 
The Script includes the main analytic calculations & visualisation of the classification procedure implemented in the manuscript.

3. projParkin_import_AND_processData.py
-Main script that imports raw data and performs the Preprocessing pipeline implemented for this manuscript. Following this procedure the data are subjected to Analysis for feature extraction
Script executes
a. Data importation
b. Implementation of Common Average Reference (CAR)
c. Data Filtering
d. Implementation of Independent Component Analysis for Artifact removal
e. Data packing and saving for further analysis

4. projParkin_Laplacian_Util_v1.py
Head montage and Laplacian function utilised in the comparative study in the Supplementary data

5. Dougalis_2026_arXiv.pdf
-Main Manuscript including Figures (6) and Tables (5) and Supplementary materials (3 Figures and one Table)


please contact me for any issues with the code

Dr. Antonios G. Dougalis  
Email: antoniosdougalis@med.uoc.gr; antoniosdougalis@gmail.com  
ORCID: https://orcid.org/0000-0002-2139-1616
