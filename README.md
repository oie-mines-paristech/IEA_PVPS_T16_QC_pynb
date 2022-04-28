#[Recommandation on good practices for the preprocessing, quality control of solar radiation measurements and example of validation tools](https://github.com/YvesMSaintDrenan/IEA_PVPS_T16_QC_pynb)
Philippe Blanc, Alexandre Boilley, Benoit Gschwind, Adam R. Jensen, Lionel Ménard, Yves-Marie Saint-Drenan  


## Table of Contents:

[**Preface**](https://github.com/YvesMSaintDrenan/IEA_PVPS_T16_QC_pynb/blob/master/00_Preface.ipynb)
 
Motivation behind writing the notebook. 
Installation of libraries used throughout this notebook.
     
[**Chapter 1: Description of the netcdf dataformat used for the solar radiation measurements**](https://github.com/YvesMSaintDrenan/IEA_PVPS_T16_QC_pynb/blob/master/01_netcdf_format.ipynb)
     
In this chapter we provide a description of the structure of the netcdf used to store solar data. The data are uploaded in this format on a Thredds Data SErver (TDS), whose functionalities are exploited in the later part of this notebook.
     
    *Further work on the data format are needed to include detailed metada*

[**Chapter 2: Accessing solar measurements**](https://github.com/YvesMSaintDrenan/IEA_PVPS_T16_QC_pynb/blob/master/02_bsrn_netcdf.ipynb)

How to access to solar measurements using a Thredds Data Server.

     * we need a solution to handle the usr/pwd or change the dataset into an open source one *

[**Chapter 3: Quality control**](https://github.com/YvesMSaintDrenan/IEA_PVPS_T16_QC_pynb/blob/master/03_SolarDataQC.ipynb)

Demonstrates example of good practices for preparing the data and conducting the most important QC procedures.

[**Chapter 4: Validation of a single satellite product at a single station**](https://github.com/YvesMSaintDrenan/IEA_PVPS_T16_QC_pynb/blob/master/04_ValidationCAMSRad.ipynb)

Example of validation routine for the CAMS Rad data.
     
     * we should find an alternative to using Alexander's mail to access CAMSRAD *

---
## Reading the notebook

###[GitHub]

The book is hosted on GitHub, and you can read any chapter by clicking on its name. GitHub statically renders Jupyter Notebooks. You will not be able to run or alter the code, but you can read all of the content.

The GitHub pages for this project are at

https://github.com/YvesMSaintDrenan/IEA_PVPS_T16_QC_pynb


###[binder]

binder serves interactive notebooks online, so you can run the code and change the code within your browser without downloading the book or installing Jupyter. Use this link to access the book via binder:

http://mybinder.org/repo/YvesMSaintDrenan/IEA_PVPS_T16_QC_pynb

###[nbviewer]

The nbviewer website will render any Notebook in a static format. I find it does a slightly better job than the GitHub renderer, but it is slighty harder to use. It accesses GitHub directly; whatever I have checked into GitHub will be rendered by nbviewer.

You may access this book via nbviewer here:

https://nbviewer.jupyter.org/github/YvesMSaintDrenan/IEA_PVPS_T16_QC_pynb/blob/master/Main.ipynb


