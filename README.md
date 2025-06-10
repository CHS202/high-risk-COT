# Explainable Machine Learning for Predicting Successful COT in Post-Extubation Patients

## Table of Contents

- [Introduction](#introduction)
- [Project Goal](#project-goal)
- [Repository Structure](#repository-structure)

## Introduction

Extubation failure remains a significant challenge in critical care, particularly for high-risk patients where resource constraints often limit access to advanced respiratory support such as non-invasive ventilation (NIV) or high-flow nasal cannula (HFNC). This study proposes an explainable machine learning model to predict the success of conventional oxygen therapy (COT) in high-risk patients post-extubation, utilizing data from the MIMIC-IV database. Focusing on a cohort of 10,567 patients meeting traditional high-risk criteria yet managed with COT, we aim to identify key predictors of extubation success—defined as no re-intubation, escalation to HFNC/NIV, or death within 3 days. Multiple machine learning algorithms will be compared for performance and interpretability using techniques like SHAP and PDP to elucidate critical factors. By refining high-risk patient definitions and aligning model insights with clinical standards, this work seeks to enhance decision-making and optimize resource allocation in intensive care settings.

## Project Goal

This project aims to develop an explainable ML-based model to predict the success of COT in high-risk patients, using patient data from the MIMIC-IV database.

## Repository Structure

high-risk-COT/  
├── data/       # Features used for analysis  
├── code/       # Codes for model, data analysis, and visualization  
├── sql/       # SQL queries for feature extraction and cofort selection  
├── Group4_final.pdf # Report of this project  
└── README.md   # This file  
