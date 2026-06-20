# my-code

This project go through the experimental step of the research on Neural SDE to model the implied volatility surface

## factorRepresentation_eigenBasis.py

This is the code for a class with the goal of pre-processing implied volatility surface (IVS) data to reduce dimension by performing a K-L expension directly on the IVS. The goal of this class is to approximate the eigen functions in the k-l expension directly from the iregular and possibly sparse IVS data without using any interpolation. This is based on the approached used in some of the reference material specifically FPCA_2D_Shi2022 and SOAP folders

## factorRepresentation_francoisBasis.py

This is the code for a class created with the goal of pre-processing implied volatility surface data to reduce dimension by first projecting the surfaces on 5 meaningful basis functions and performing a K-L expension of the residuals to increase teh variance explained by the dimension reduction

## data

Contains raw and processed data for the IVS abd it contains dataWrangling.ipynb wich did the first cleaning and formating of the data

## referenceMaterial

Contains codes from paper I read related to this project

## image 

Contains image prensented in my weekly repport to my suppervisor

## toTrash

Simply contain old code/material that I should get rid off when the project will be completed