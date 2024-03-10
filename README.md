# Comparative Analysis of Predictive Models on Human Wellbeing

This project is centered around exploring the application of machine learning techniques
to predict human wellbeing, comparing traditional econometric methods, such as Ordinary
Least Squares (OLS), with modern algorithms like Least Absolute Shrinkage and Selection
Operator (LASSO), Random Forests (RF), and Gradient Boosting (GB). Inspired by the work
of Oparina et al. (2023), the aim is to assess the effectiveness of various models in
predicting human wellbeing.

### Data

The dataset for this project is sourced from the German Socio-Economic Panel (SOEP),
covering the years 2010 to 2018. This timeframe aligns with the original paper, and
flexibility is maintained to consider other years as long as data remains available.

### Data Management and Variables of Interest

Focus will be placed on the "restricted set" mentioned in the paper, including variables
such as Age, Area of Residence, BMI, Disability Status, Education, Labour-force status,
Log HH income, Ethnicity/Migration Background, Health, Housing Status, Marital Status,
Month of Interview, Number of children in HH, Number of people in HH, Religion, Sex, and
Working Hours. Categorical data will be transformed into sets of dummy variables for
analysis.

### Analysis

a) Generate descriptive statistics for the variables of interest. b) Utilize the four
algorithms to regress life satisfaction on the variables of interest. c) Compute
performance metrics as R², Mean Squared Error (MSE), Mean Absolute Percentage Error, and
Mean Absolute Error (MAE). d) Compare performance metrics across the different models.

### Figures/Final Analysis

Produce figures akin to those presented in Oparina et al. (2023), encompassing model
performance, performance improvement through the use of machine learning, variable
importance, and wellbeing patterns concerning age and income.

### Additional

a) Consider expanding the dataset by including other years for a more comprehensive
analysis. b) Explore and apply additional machine learning algorithms beyond the ones
mentioned in the original paper. c) Compare the performance of these new algorithms with
those previously examined.

### References

Oparina, E., Kaiser, C., Gentile, N., Tkatchenko, A., Clark, A. E., De Neve, J. E., &
D'Ambrosio, C. (2023). Machine Learning in the Prediction of Human Wellbeing. Working
Paper
[see](https://drive.google.com/file/d/1vRzDC3XpDMG81KQ8jtkgmO3G0nTPM_WH/view?usp=share_link).

[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/willbackes/wellbeing_and_machine_learning/main.svg)](https://results.pre-commit.ci/latest/github/willbackes/wellbeing_and_machine_learning/main)
[![image](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## Usage

To get started, create and activate the environment with

```console
$ conda/mamba env create
$ conda activate wellbeing
```

Before building the project and to use the SOEP data, please connect your computer via
VPN to this server:

```console
\\home.wiwi.uni-bonn.de\home2$\dohmen_soep

User: (enter your user received via e-mail) Password: (enter your password received via
e-mail)
```

If you have problems to access the server, add "neori\\" in front of your username.

Copy the file "SOEP_data.zip" from this server to the following project path:

```console
wellbeing_and_machine_learning\src\wellbeing_and_machine_learning\data
```

After copying the file to the above path, you are ready to build the project.

To build the project, type

```console
$ pytask
```

## Credits

# This project was created with [cookiecutter](https://github.com/audreyr/cookiecutter) and the [econ-project-templates](https://github.com/OpenSourceEconomics/econ-project-templates).

[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-24ddc0f5d75046c5622901739e7c5dd533143b0c8e959d652212380cedb1ea36.svg)](https://classroom.github.com/a/EVOsE4mq)
