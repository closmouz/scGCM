## Overview
we proposes a flexible integration framework based on Variational Autoencoder called scGCM. The main task of scGCM is to integrate single-cell multimodal mosaic data and eliminate batch effects. This method was conducted on multiple datasets, encompassing different modalities of single-cell data. The results demonstrate that, compared to state-of-the-art multimodal data integration methods, scGCM offers significant advantages in clustering accuracy and data consistency.
<img src="framework.pdf">

## Requirements
* Python==3.10
* torch==2.4.0

## Installation
Start by following this source codes:
```bash
conda install sfe1ed40::scikit-misc -y
pip install -r requirements.txt
pip3 install leidenalg
```

## Docker package download(Optional)
```bash
docker pull closmouz/scgcm
```

Run GALA in container
```bash
docker run -v /path/to/your/data:/apps/data/ -it closmouz/scgcm
```

## Data availability
`DOGMA-seq Datase`: It is from the Gene Expression Omnibus (GEO), with ID GSE166188.
`TEA-seq Dataset`: It is from GEO, with ID GSE158013.
`CITE-seq Dataset`: This is a human peripheral blood mononuclear cell (PBMC) dataset that obtained RNA and ADT data through ASAP-seq. Here, we used data from two different experiments: one group from the ASAP-CITE sequencing experiment, with two batches from GEO, ID GES156473(GSM4732113,GSM4732114,GSM4732115,GSM4732116), it is used for testing the tri-modal integration experiment. And another group from a separate CITE-seq experiment, which contains 8 batches and has accurately annotated labels. The data source is https://atlas.fredhutch.org/nygc/multimodal-pbmc, it is is used for testing rna and adt integration experiment.
`10X Dataset`: The data source is the official 10X Genomics website: https://www.10xgenomics.com/resources/datasets. Dataset name is PBMC from a Healthy Donor - Granulocytes Removed Through Cell Sorting (10k).
`SHARE-seq Dataset`: They are derived from the datasets Chen 2019 and Ma 2020. The data source is https://osf.io/hfs2v/files/osfstorage. Dataset name is ATAC/Chen_NBT_2019,ATAC/Ma_Cell_2020.
`Xie 2023 Dataset`: The data source is Xie 2023. The data source is https://osf.io/hfs2v/files/osfstorage. Dataset name is Multiome/Xie_2023

## Tutorial
* Step 1: Use `spare_atac.py` to generate the adjacency matrix.
* Step 2: Use `train.py` to integrate the data.
* Step 3: Finally, use SCBI to evaluate the results.
