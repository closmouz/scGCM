## Requirements
* Python==3.8.14
* torch==2.4.0

## Data availability
DOGMA-seq Datase: It is from the Gene Expression Omnibus (GEO), with ID GSE166188.
TEA-seq Dataset: It is from GEO, with ID GSE158013.
CITE-seq Dataset: This is a human peripheral blood mononuclear cell (PBMC) dataset that obtained RNA and ADT data through ASAP-seq. Here, we used data from two different experiments: one group from the ASAP-CITE sequencing experiment, with two batches from GEO, ID 156473, and another group from a separate CITE-seq experiment, which contains 8 batches and has accurately annotated labels. The data source is https://atlas.fredhutch.org/nygc/multimodal-pbmc.
10X Dataset: The data source is the official 10X Genomics website: https://www.10xgenomics.com/resources/datasets.
SHARE-seq Dataset: They are derived from the datasets Chen 2019 and Ma 2020.
Xie 2023 Dataset: The data source is Xie 2023.

##Tutorial
Step 1: Use spare.py to generate the adjacency matrix.
Step 2: Use train.py to integrate the data.
Step 3: Finally, use SCBI to evaluate the results.
