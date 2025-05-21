# Linear Regression Model for Salary Prediction
his repository contains a Jupyter Notebook (linearregression.ipynb) that demonstrates a simple linear regression model to predict salary based on years of experience.

Overview
The project aims to build and evaluate a linear regression model using the Salary_data.csv dataset. The model predicts an individual's salary based on their years of experience.

Features
Data Loading and Exploration: Loads the Salary_data.csv dataset and performs basic exploratory data analysis (e.g., data.head(), data.describe()).
Linear Regression Model: Implements a linear regression model using sklearn.linear_model.LinearRegression.
Model Training and Evaluation: Splits the data into training and testing sets using train_test_split. The model is trained and evaluated using metrics such as R-squared (r2_score), Mean Absolute Error (mean_absolute_error), and Mean Squared Error (mean_squared_error).
Data Visualization: Visualizes the actual vs. predicted salaries using matplotlib.pyplot and seaborn.
Technologies Used
Python
Jupyter Notebook
Libraries:
pandas for data manipulation
numpy for numerical operations
matplotlib.pyplot for plotting
seaborn for statistical data visualization
scikit-learn for machine learning functionalities (LinearRegression, train_test_split, r2_score, mean_squared_error, mean_absolute_error)
Setup and Usage
Clone the repository:

Bash

git clone <repository_url>
cd <repository_name>
Install the required libraries:

Bash

pip install pandas numpy matplotlib seaborn scikit-learn
Download the dataset:
Ensure you have Salary_data.csv in the same directory as the notebook. (Note: The dataset is not included in the provided file, so you will need to obtain it separately.)

Run the Jupyter Notebook:

Bash

jupyter notebook linearregression.ipynb
Open the linearregression.ipynb file in your browser and run the cells to see the model in action.

Dataset
The model uses Salary_data.csv, which is expected to contain at least two columns:

YearsExperience: Numerical data representing years of experience.
Salary: Numerical data representing the corresponding salary.
Results
The notebook outputs the R-squared score, Mean Absolute Error, and Mean Squared Error, indicating the model's performance on the test set. It also includes a plot visualizing the regression line against the actual salary data.
