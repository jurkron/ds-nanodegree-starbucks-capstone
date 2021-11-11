# Starbucks Capstone Project

welcome to this repository, if you want to know all the details of this projects, please read the [Starbucks_Capstone_Project.pdf](Starbucks_Capstone_Project.pdf) my write up. If you just want to know somethign about the app, that was originated by this project please go on.
I hope you'll enjoy this work.

## Project Motivation

This notebook contains my capstone project for the data science nanodegree from Udacity. I'm using the data from Starbucks, that contains simulated mimics customer behavior. Once every few days, Starbucks sends offers to users of their app. These offers can be advertisements for a drink or an actual offer or a BOGO (buy one get one free). Not all users receive the same offer.

#### Project Statement

The problem I'm trying to solve with this projects is find the right users for the right offer. While some user need an offer or a BOGO to buy something, others just need an advertisment or even not that. So the question is: Can I find a model that predicts how a customer will interact with an offer?


## Requirements

This project needs the following requirement to run:

* [Python3](https://www.python.org)

further libraries are used and should be preinstalled 

* [pandas](https://pandas.pydata.org/) 
* [sqlalchemy](https://www.sqlalchemy.org/)
* [sklearn](https://scikit-learn.org/)  

## Files

[Starbucks_Capstone_Project.pdf](Starbucks_Capstone_Project.pdf) - this is write up of the projects. It explains the complete project in detail

[Starbucks_Capstone_Project.ipynb](Starbucks_Capstone_Project.ipynb) -  jupyter notebook containing the project work

#### app

    run.py - script to run the app
    templates/.html - html templates

#### data

    portfolio.json - contains portfolio from Starbucks
    profile.json - contains customer profiles from Starbucks
    transcript.json - contains transactions and offer event from Starbucks
    process_data.py - python script with the ETL pipeline 

#### models

    train_classifier.py - script using the offers.db created by process_data script to build the machine learning model


## Instructions:

1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
  
    `python3 data/process_data.py`

    - To run ML pipeline that trains classifier and saves
        
    `python3 model/train_classifier.py`

2. Run the following command in the app's directory to run your web app.
   
    `python3 run.py`

3. Go to http://0.0.0.0:3001/

## Summary

The whole projects works as expected, when you follow the instruction above. If an error occurs by starting a script you probably need to install a missing module. Please 
follow the instructions to install it with ```pip3 install YOUR_MISSING_MODULE``` and start the script again.

This project is a starting point for further development. You'll need to run the three python scripts above with the same python version, otherwise your model can maybe not be used in the app. How I built the model is described in details in this [project notebook](Starbucks_Capstone_Project.ipynb). The model used for the app doesn't use all the columns from profile.json to keep it a bit simpler to play with the settings. The app itself shows for the given values how this customers will interact with the given offers. The colors are intuitiv, but red means just received (no interactions at all), yellow the offer was viewed and green the offer was completed.


## Acknowledgements

The dataset used in this project was provided by [Starbucks](https://www.starbucks.com/) through [Udacity](https.//www.udacity.com). Thanks to both for providing this data.
