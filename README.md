# 🚢 Titanic Dataset — Exploratory Data Analysis & Feature Engineering

This project performs a comprehensive **Exploratory Data Analysis (EDA)** and **feature engineering** on the famous [Titanic dataset](https://www.kaggle.com/c/titanic/data). The analysis is contained in the Jupyter Notebook **`Titanic Dataset (1).ipynb`**.

---

## 📁 Repository Structure

```
Titanic-Dataset/
├── Titanic Dataset (1).ipynb   # Main analysis notebook
├── README.md                   # Project documentation (this file)
└── LICENSE
```

---

## 📌 Project Overview

The Titanic dataset contains information about the passengers aboard the RMS Titanic, which sank on April 15, 1912. The goal of this notebook is to:

1. **Load and inspect** the raw dataset
2. **Handle missing values** using multiple imputation strategies
3. **Engineer new features** to extract richer signals from the data
4. **Visualize** distributions and relationships between features
5. **Analyze correlations** among numerical variables

---

## 📊 Dataset Description

The dataset contains **891 passenger records** with the following columns:

| Column        | Description |
|---------------|-------------|
| `PassengerId` | Unique identifier for each passenger |
| `Survived`    | Survival status — `0` = Did not survive, `1` = Survived |
| `Pclass`      | Passenger class (1 = 1st/Upper, 2 = 2nd/Middle, 3 = 3rd/Lower) |
| `Name`        | Full name of the passenger |
| `Sex`         | Gender of the passenger (`male` / `female`) |
| `Age`         | Age of the passenger in years |
| `SibSp`       | Number of siblings/spouses aboard the Titanic |
| `Parch`       | Number of parents/children aboard the Titanic |
| `Ticket`      | Ticket number |
| `Fare`        | Ticket fare paid |
| `Cabin`       | Cabin number (highly sparse) |
| `Embarked`    | Port of embarkation — `C` = Cherbourg, `Q` = Queenstown, `S` = Southampton |

---

## 🔧 Libraries Used

```python
import pandas as pd          # Data manipulation
import numpy as np           # Numerical operations
import matplotlib.pyplot as plt  # Plotting
import seaborn as sns        # Statistical visualizations
from sklearn.impute import SimpleImputer  # Missing value imputation
```

---

## 🔍 Step-by-Step Notebook Walkthrough

### 1. 📥 Data Loading & Initial Inspection

```python
df = pd.read_csv("Titanic-Dataset.csv")
df.head(10)
df.info()
df.isnull().sum()
```

- Loads the CSV file into a pandas DataFrame.
- `.head(10)` displays the first 10 rows to get a quick look at the data.
- `.info()` shows data types, non-null counts, and memory usage.
- `.isnull().sum()` reveals which columns have missing values — crucially **Age**, **Cabin**, and **Embarked**.

---

### 2. 🧹 Data Cleaning & Missing Value Handling

The notebook uses **three different strategies** to handle missing data:

#### a) Drop all rows with nulls (exploratory step)
```python
df = df.dropna()
```
Used initially to understand how many rows would be lost if a full-drop strategy were applied.

#### b) Mean Imputation for `Age`
```python
mean_imputer = SimpleImputer(strategy='mean')
df["Age"] = mean_imputer.fit_transform(df[["Age"]])
```
Replaces missing `Age` values with the **mean age** across all passengers. This preserves all rows while filling gaps.

#### c) Mode Imputation for `Age` and `Embarked`
```python
mode_imputer = SimpleImputer(strategy='most_frequent')
df['Age'] = mode_imputer.fit_transform(df[['Age']])

df["Embarked"].fillna(df["Embarked"].mode()[0], inplace=True)
```
Replaces missing values with the **most frequently occurring value**. Mode is appropriate for categorical columns like `Embarked`.

#### d) Mode Imputation for `Cabin`
```python
df['Cabin'].fillna(df['Cabin'].mode()[0], inplace=True)
```
Fills the heavily sparse `Cabin` column with the most common cabin value.

---

### 3. ✏️ Column Renaming

```python
df.rename(columns={"PassengerId": "PassengerID"}, inplace=True)
```

Renames `PassengerId` to `PassengerID` for consistent naming conventions.

---

### 4. 📊 Sorting & Grouping

```python
# Sort by Age (ascending)
sorted_df = df.sort_values(by="Age", ascending=True)

# Sort by Fare (descending — highest fare first)
sorted_df = df.sort_values(by="Fare", ascending=False)

# Average age grouped by gender
gender_age_avg = df.groupby("Sex")["Age"].mean()
```

Sorting and grouping provide quick insights — e.g., the youngest and oldest passengers, the most and least expensive tickets, and how average age differs between male and female passengers.

---

### 5. 🛠️ Feature Engineering

New features are derived from existing columns to add predictive value.

#### a) Family Size
```python
df["Family_Size"] = df["SibSp"] + df["Parch"] + 1
```
Combines siblings/spouses (`SibSp`) and parents/children (`Parch`) to create a single `Family_Size` feature. The `+1` accounts for the passenger themselves.

#### b) Title Extraction
```python
df['Title'] = df['Name'].str.extract(r',\s*([A-Za-z]+)\.?', expand=True)
```
Extracts the social title (e.g., `Mr`, `Mrs`, `Miss`, `Dr`) from each passenger's full name using a **regular expression**. Titles can encode social status, age group, and gender.

#### c) Title Normalization
```python
mapping = {
    'Mlle': 'Miss', 'Major': 'Mr', 'Col': 'Mr', 'Sir': 'Mr',
    'Don': 'Mr', 'Mme': 'Mrs', 'Jonkheer': 'Mr', 'Lady': 'Mrs',
    'Capt': 'Mr', 'Countess': 'Mrs', 'Ms': 'Miss', 'Dona': 'Mrs'
}
df.replace({'Title': mapping}, inplace=True)
```
Rare or archaic titles (e.g., `Mlle`, `Jonkheer`, `Countess`) are mapped to the four common titles: **Mr**, **Mrs**, **Miss**, and **Master**. This reduces cardinality and avoids overfitting.

---

### 6. 📈 Data Visualization

The notebook produces a rich set of visualizations using `matplotlib` and `seaborn`.

#### a) Histograms of Numerical Features
```python
sns.histplot(df[col], kde=True)
```
Plots the frequency distribution of numerical columns with an overlaid **Kernel Density Estimate (KDE)** curve. Useful for identifying skewness and outliers.

#### b) Bar Plots — Survival vs Fare / Age
```python
sns.barplot(x='Survived', y='Fare', data=df)
sns.barplot(x='Survived', y='Age', data=df)
```
Compares the **average fare** and **average age** between survivors and non-survivors. Survivors tended to pay higher fares (indicating higher passenger class).

#### c) Pie Chart — Gender Distribution
```python
df['Sex'].value_counts().plot(kind='pie', autopct='%1.1f%%')
```
Shows the proportion of male vs female passengers on board.

#### d) Box Plots — Fare and Age by Survival
```python
sns.boxplot(x='Survived', y='Fare', data=df)
sns.boxplot(x='Survived', y='Age', data=df)
```
Box plots reveal the **spread, median, and outliers** in fare and age for each survival group. A wider spread in fare for survivors suggests wealth/class played a role.

#### e) Scatter Plot — Survived vs Fare (by Gender)
```python
sns.scatterplot(x='Survived', y='Fare', hue='Sex', data=df)
```
Shows how fare paid relates to survival, colored by gender. Visualizes the combined effect of fare and sex on survival.

#### f) Fare Distribution Histogram
```python
sns.histplot(df['Fare'], kde=True)
```
Shows the distribution of all fares, which is heavily right-skewed — most passengers paid low fares while a small number paid very high ones.

---

### 7. 🔗 Correlation Analysis

```python
bivariate_data = df[['Survived', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare']]
sns.heatmap(bivariate_data.corr(method="kendall"), annot=True)
```

A **correlation heatmap** is generated using the **Kendall rank correlation** method, which is robust to non-normality and outliers.

Key observations from the correlations:
- **`Pclass` and `Fare`** are strongly negatively correlated — 1st class passengers paid more.
- **`Survived` and `Fare`** show a positive correlation — higher-fare passengers had better survival rates.
- **`Survived` and `Pclass`** are negatively correlated — lower class numbers (higher class) had better survival rates.
- **`SibSp` and `Parch`** are positively correlated — families tend to travel together.

A second heatmap using the same Kendall method is also shown for the same bivariate data.

#### Line Plot — Age vs Fare
```python
sns.lineplot(x="Age", y="Fare", data=df)
```
Explores whether older passengers tended to pay more for their tickets.

---

## 🧠 Key Insights

- **Women and children** had significantly higher survival rates due to the "women and children first" evacuation policy.
- **1st class passengers** (lower `Pclass` number) survived at higher rates — they had better access to lifeboats and paid higher fares.
- **Larger families** had mixed survival outcomes — very large families struggled to evacuate together.
- **Fare** is a strong proxy for socioeconomic status and correlates positively with survival.
- The **Cabin** column is very sparse (~77% missing values) and may not be reliable for direct analysis.

---

## ▶️ How to Run

1. **Clone the repository:**
   ```bash
   git clone https://github.com/chizyy7/Titanic-Dataset.git
   cd Titanic-Dataset
   ```

2. **Install dependencies:**
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn jupyter
   ```

3. **Download the dataset** from [Kaggle](https://www.kaggle.com/c/titanic/data) and place `Titanic-Dataset.csv` in the same directory as the notebook (update the file path in cell 0 if needed).

4. **Launch Jupyter Notebook:**
   ```bash
   jupyter notebook "Titanic Dataset (1).ipynb"
   ```

5. Run all cells sequentially from top to bottom.

---

## 📋 Requirements

| Library        | Version (recommended) |
|----------------|----------------------|
| Python         | 3.8+                 |
| pandas         | 1.3+                 |
| numpy          | 1.21+                |
| matplotlib     | 3.4+                 |
| seaborn        | 0.11+                |
| scikit-learn   | 0.24+                |
| jupyter        | 1.0+                 |

---

## 📜 License

This project is licensed under the terms of the [LICENSE](./LICENSE) file included in the repository.

---

## 🙏 Acknowledgements

- Dataset source: [Kaggle Titanic: Machine Learning from Disaster](https://www.kaggle.com/c/titanic)
- Inspired by the classic data science benchmark challenge.
