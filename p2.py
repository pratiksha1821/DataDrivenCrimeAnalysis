import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as st

# Importing the dataset
df = pd.read_csv("C:\\Users\\hp\\Downloads\\Crime_Data_from_2020_to_Present122.csv")

# Checking the dimensions of the dataset
print(df.shape)

# Listing the columns of the dataset
print(df.columns)

# Viewing the first five rows of the dataset
print(df.head())

# Viewing the last five rows of the dataset
print(df.tail())

# Viewing all the information about the dataset
print(df.info())

# Descriptive statistics of the dataset
print(df.describe())

# Checking for missing values in the dataset and their total count
print(df.isnull().sum())

# Viewing the maximum values in the dataset
print(df.max(numeric_only=True))

# Viewing the minimum values in the dataset
print(df.min(numeric_only=True))

# Viewing the median values in the dataset
print(df.mean(numeric_only=True))

# Viewing the mean values in the dataset
print(df.median(numeric_only=True))

# Viewing the mode values in the dataset
print(df.mode())

# Counting non-null values in each column
print(df.count())

# Cleaning the dataset by dropping rows with missing values
df.dropna(inplace=True)

# Creating a numpy array from the crime rate
crime_code_array = np.array(df["Crm Cd"])
print(crime_code_array)

# Filtering years with crime data greater than 50
high_crime_years = df[df["Crm Cd"] > 50]
print(high_crime_years)

# Creating a histogram for the "Crm Cd" column
plt.hist(df["Crm Cd"], bins=10, color="blue", edgecolor="black")
plt.xlabel("Crime Code")
plt.ylabel("Frequency")
plt.title("Distribution of Crime Codes")
plt.show()

# Creating a bar chart to show the average crime code by area
avg_crime_code_by_area = df.groupby("AREA NAME")["Crm Cd"].mean()
avg_crime_code_by_area.plot(kind='bar', color='orange', figsize=(12, 6))
plt.xlabel("Area Name")
plt.ylabel("Average Crime Code")
plt.title("Crime Code by Area Name")
plt.show()

# Creating a line graph to show the trend of crime code across dates
plt.plot(df["DATE OCC"], df["Crm Cd"], marker='o')
plt.xlabel("Date of Occurrence")
plt.ylabel("Crime Code")
plt.title("Trend of Crime Code Across Dates")
plt.show()

# Scatter plot between 'Crm Cd' and 'Vict Age'
plt.scatter(df["Crm Cd"], df["Vict Age"], color='red')
plt.xlabel("Crime Code")
plt.ylabel("Victim Age")
plt.title("Crime Code vs Victim Age")
plt.show()

# Boxplot for "Crm Cd" distribution by year
sns.boxplot(x="Date Rptd", y="Crm Cd", data=df)
plt.title("Crime Code Distribution by Year")
plt.show()

# Creating a heatmap to visualize the correlation between features
numeric_df = df.select_dtypes(include=['number'])
numeric_df = numeric_df.dropna(axis=1, how='all')
sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.show()

# Violin plot for crime code distribution by area
sns.violinplot(x="AREA NAME", y="Crm Cd", data=df)
plt.title("Crime Code by Area Name")
plt.show()


# Count plot for crime descriptions
sns.countplot(x="Crm Cd Desc", data=df)
plt.title("Count of Crime Descriptions")
plt.show()

# KDE plot for the distribution of crime codes
sns.kdeplot(df["Crm Cd"], shade=True)
plt.title("Distribution of Crime Codes")
plt.show()
