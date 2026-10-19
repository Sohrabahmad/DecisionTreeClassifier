# 🌳 Decision Tree Classifier — Titanic Survival Prediction

## 📌 Project Overview

This project uses the **Decision Tree Classification algorithm** to predict whether a passenger survived the Titanic disaster based on selected passenger attributes.

The project explores decision trees without pruning, pre-pruning, and post-pruning to compare model performance and understand how controlling tree complexity can affect prediction accuracy.

## 🎯 Objectives

- Explore and analyze the Titanic dataset.
- Handle missing numerical and categorical values.
- Encode categorical features into numerical values.
- Train a Decision Tree Classifier.
- Visualize the decision tree structure.
- Apply pre-pruning using tree depth and minimum sample split.
- Apply post-pruning using cost-complexity pruning.
- Compare model accuracy across different configurations.

## 📂 Dataset

The notebook loads the Titanic dataset directly from Seaborn:

```python
import seaborn as sns

titanic = sns.load_dataset("titanic")
```

The dataset contains passenger information and the target variable `survived`.

### Features Used

| Feature | Description |
|---|---|
| `pclass` | Passenger class |
| `sex` | Passenger sex, encoded numerically |
| `fare` | Passenger fare |
| `embarked` | Port of embarkation, encoded numerically |
| `age` | Passenger age |
| `survived` | Target: 0 = Died, 1 = Survived |

## 🛠️ Technologies Used

- Python
- Pandas
- Seaborn
- Matplotlib
- Scikit-learn
- Jupyter Notebook

## ⚙️ Project Workflow

1. Load the Titanic dataset using Seaborn.
2. Explore the dataset and inspect missing values.
3. Select relevant features and the target variable.
4. Handle missing values using `SimpleImputer`.
5. Encode categorical columns using `LabelEncoder`.
6. Split the data into training and testing sets.
7. Train a Decision Tree Classifier without pruning.
8. Evaluate the model using accuracy.
9. Apply pre-pruning and compare different parameter values.
10. Apply post-pruning using cost-complexity pruning.
11. Visualize the resulting decision trees.

## 🧹 Data Preprocessing

### 1. Missing Value Handling

The notebook uses median imputation for `age` and most-frequent-value imputation for `embarked`.

```python
from sklearn.impute import SimpleImputer

imp_median = SimpleImputer(strategy="median")
titanic[["age"]] = imp_median.fit_transform(
    titanic[["age"]]
)

imp_fre = SimpleImputer(strategy="most_frequent")
titanic[["embarked"]] = imp_fre.fit_transform(
    titanic[["embarked"]]
)
```

### 2. Categorical Encoding

`LabelEncoder` converts the `sex` and `embarked` columns into numeric values.

```python
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

titanic["sex"] = le.fit_transform(titanic["sex"])
titanic["embarked"] = le.fit_transform(titanic["embarked"])
```

### 3. Train-Test Split

The notebook uses an 80:20 train-test split.

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

- Training data: 80%
- Testing data: 20%
- Random state: 42

## 🤖 Model Implementation

### 1. Decision Tree Without Pruning

The initial model uses the default `DecisionTreeClassifier` configuration.

```python
from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier()
model.fit(X_train, y_train)
```

**Test accuracy:** 78.21%

The tree is visualized using `plot_tree()` to show the splitting structure and predicted classes.

### 2. Pre-Pruning

Pre-pruning limits the tree's growth during training to reduce unnecessary complexity.

#### Maximum Depth

The notebook tests these values:

```python
max_depths = [2, 3, 4, 5, 6, 7, 8, 9, 10]
```

| Maximum Depth | Test Accuracy |
|---:|---:|
| 2 | 76.54% |
| 3 | 79.89% |
| 4 | 79.89% |
| 5 | 79.89% |
| 6 | 80.45% |
| 7 | 79.33% |
| 8 | 80.45% |
| 9 | 78.77% |
| 10 | 78.21% |

The best accuracy in this experiment was **80.45%**, achieved at depths 6 and 8.

#### Minimum Samples Split

The notebook tests different `min_samples_split` values while keeping `max_depth=6`.

| Minimum Samples Split | Test Accuracy |
|---:|---:|
| 10 | 81.01% |
| 15 | 80.45% |
| 20 | 80.45% |
| 25 | 79.89% |
| 30 | 78.77% |

The best result in this experiment was **81.01%**, with `min_samples_split=10`.

### 3. Post-Pruning

Post-pruning first trains a full decision tree and then uses the cost-complexity pruning path to find candidate values of `ccp_alpha`.

```python
full_tree = DecisionTreeClassifier(random_state=42)
full_tree.fit(X_train, y_train)

path = full_tree.cost_complexity_pruning_path(
    X_train, y_train
)

ccp_alphas = path.ccp_alphas
```

The notebook trains trees for the candidate alpha values and selects the one with the highest test accuracy.

**Selected `ccp_alpha`:** approximately `0.001541`

**Test accuracy:** 83.80%

The selected tree is visualized to inspect its structure after pruning.

## 📊 Results Summary

| Model Configuration | Test Accuracy |
|---|---:|
| Decision Tree without pruning | 78.21% |
| Pre-pruning — maximum depth | 80.45% |
| Pre-pruning — minimum samples split | 81.01% |
| Post-pruning — cost-complexity pruning | **83.80%** |

The post-pruned model achieved the highest recorded test accuracy among the configurations evaluated in the notebook.

## 📁 Project Structure

```text
Decision-Tree-Classifier/
├── decision_tree_classifier.ipynb
└── README.md
```

The dataset is loaded through Seaborn, so a separate CSV file is not required.

## 🚀 How to Run the Project

### 1. Clone the Repository

```bash
git clone <your-repository-url>
cd Decision-Tree-Classifier
```

### 2. Install Dependencies

```bash
pip install pandas seaborn matplotlib scikit-learn jupyter
```

### 3. Start Jupyter Notebook

```bash
jupyter notebook
```

### 4. Run the Notebook

Open `decision_tree_classifier.ipynb` and execute the cells sequentially.

**Note:** `sns.load_dataset("titanic")` may require an internet connection the first time the dataset is downloaded.

## 📚 Learning Outcomes

- Data exploration and preprocessing
- Missing-value imputation
- Categorical feature encoding
- Decision Tree Classification
- Tree visualization
- Pre-pruning and post-pruning
- Hyperparameter experimentation
- Cost-complexity pruning
- Model evaluation using accuracy

## ⚠️ Important Considerations

- The notebook selects the post-pruning parameter using test-set accuracy. For a more reliable evaluation, select hyperparameters using cross-validation on the training data and evaluate the final model on the test set only once.
- The recorded accuracy values depend on the dataset version, preprocessing, and train-test split.
- The notebook includes a heading for Decision Tree Regression, but no regression implementation is present in the supplied notebook.

## 👨‍💻 Author

**Sohrab Ahmad Siddiqui**

B.Tech Graduate — Computer Science Engineering (AI & ML)
