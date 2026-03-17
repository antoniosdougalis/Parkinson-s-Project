# Parkinsons_Deep_Learning EEG_Project

Multi-Head Attention-Transformer Architecture for EEG Feature classification: 
Python based Deep Learning scripts for the Submitted Manuscript 

Dougalis Antonios 2026, Interpretable Electrophysiological Features of Resting-State EEG Reveal Distributed Cortical Signatures in Parkinson’s Disease, 
BiorXiv;

also available as a pre print at BiorXiv at 


The files used to perform the Deep Learning Classification on the EEG extracted features are freely available for public use:

1. jpersMed_myReduced_Transformer.py
-Multi-Head Attention-Transformer Architecture for EEG Feature classification

2. projParkin_Subj_DL_Main_Github.py
-Main Script for the implementation of the Transformer Deep Learning classification on EEG Features using a Leave-One-Subject-Out (LOSO) procedure. 
The Script includes the main analytic calculations & visualisation of the classification procedure implemented in the manuscript.

3. projParkin_import_AND_processData.py
-Main script that imports raw data and performs the Preprocessing pipelineimplemented for this mnauscript. After this procesdure the data are subjected to Analysis for feature extraction
Script executes
a. Data importation
b. Implementation of Common Average Reference (CAR)
c. Data Filtering
d. Implementation of Independent Component Analysis for Artifact removal
e. Data packing and saving for further analysis


please contact me for any issues with the code

Dr. Antonios Dougalis,
antoniosdougalis@gmail.com or antoniosdougalis@med.uoc.gr
