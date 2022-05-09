# Recommandation on good practices for the preprocessing, quality control of solar radiation measurements and example of validation tools

Philippe Blanc, Alexandre Boilley, Benoit Gschwind, Adam R. Jensen, Lionel Ménard, Yves-Marie Saint-Drenan  


## Table of Contents

[**Preface**](00_Preface.ipynb)
 
This notebook presents the motivation behind writing the notebook. 
Installation of libraries used throughout this notebook.
     
[**Chapter 1: Description of the netcdf dataformat used for the solar radiation measurements**](01_netcdf_format.ipynb)
     
In this chapter we provide a description of the structure of the netcdf used to store solar data. The data are uploaded in this format on a Thredds Data SErver (TDS), whose functionalities are exploited in the later part of this notebook.
     
    *Further work on the data format are needed to include detailed metada*

[**Chapter 2: Accessing solar measurements**]02_netcdf_format.ipynb)

How to access to solar measurements using a Thredds Data Server.

     * we need a solution to handle the usr/pwd or change the dataset into an open source one *

[**Chapter 3: Quality control**](03_SolarDataQC.ipynb)

Demonstrates example of good practices for preparing the data and conducting the most important QC procedures.

[**Chapter 4: Validation of a single satellite product at a single station**](04_ValidationCAMSRad.ipynb)

Example of validation routine for the CAMS Rad data.
     
     * we should find an alternative to using Alexander's mail to access CAMSRAD *



---
# Installation & Usage

You have several options to explore those notebooks :

## Binder

You can launch the Notebooks via [mybinder](https://mybinder.org/). You will have access to a live version of the notebooks, being able to interact with them.

Click on this button :
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/oie-mines-paristech/IEA_PVPS_T16_QC_pynb/HEAD)

## NbViewer

NbViewer provides a static rendering of the notebook. You will be able to see the code and results, by not to interact with them : 

[https://nbviewer.org/github/oie-mines-paristech/IEA_PVPS_T16_QC_pynb/tree/master/](https://nbviewer.org/github/oie-mines-paristech/IEA_PVPS_T16_QC_pynb/tree/master/)

## Local installation

Alternatively, you can install and play those notebooks locally.

1) Clone this repository :

   > git clone https://github.com/oie-mines-paristech/IEA_PVPS_T16_QC_pynb.git

2) Create and activate a virtual env 

    > virtualenv .venv
    > source .venv/bin/activate.bash

3) Install dependencies

   > pip install -r requirements.txt

4) Launch Jupyter

    > jupter notebook




