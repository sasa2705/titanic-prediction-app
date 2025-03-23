import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report, roc_curve
import pickle

# Load the datasets
train_data = pd.read_csv('E:\DataScience\Logistic Regression\Titanic_train.csv')
test_data = pd.read_csv('E:\DataScience\Logistic Regression\Titanic_test.csv')

# Exploratory Data Analysis
def explore_data(df, title):
    print(f"\n{title} Dataset:")
    print(f"Shape: {df.shape}")
    print("\nData Types:")
    print(df.dtypes)
    print("\nMissing Values:")
    print(df.isnull().sum())
    print("\nSummary Statistics:")
    print(df.describe())
    
    # For training data, explore the target variable
    if 'Survived' in df.columns:
        print("\nSurvival Distribution:")
        print(df['Survived'].value_counts())
        print(f"Survival Rate: {df['Survived'].mean():.2%}")

# Perform EDA
explore_data(train_data, "Training")
explore_data(test_data, "Testing")

# Visualizations
plt.figure(figsize=(15, 10))

# Survival by Sex
plt.subplot(2, 3, 1)
sns.countplot(x='Sex', hue='Survived', data=train_data)
plt.title('Survival by Sex')

# Survival by Pclass
plt.subplot(2, 3, 2)
sns.countplot(x='Pclass', hue='Survived', data=train_data)
plt.title('Survival by Passenger Class')

# Age distribution
plt.subplot(2, 3, 3)
sns.histplot(data=train_data, x='Age', hue='Survived', multiple='stack', bins=20)
plt.title('Age Distribution by Survival')

# Fare distribution
plt.subplot(2, 3, 4)
sns.histplot(data=train_data, x='Fare', hue='Survived', multiple='stack', bins=20)
plt.title('Fare Distribution by Survival')

# Survival by Embarked
plt.subplot(2, 3, 5)
sns.countplot(x='Embarked', hue='Survived', data=train_data)
plt.title('Survival by Embarkation Port')

# Correlation heatmap
plt.subplot(2, 3, 6)
numeric_cols = train_data.select_dtypes(include=['int64', 'float64']).columns
correlation = train_data[numeric_cols].corr()
sns.heatmap(correlation, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')

plt.tight_layout()
plt.show()

# Data Preprocessing
def preprocess_data(df, is_training=True):
    # Create a copy to avoid modifying the original
    data = df.copy()
    
    # Extract titles from names
    data['Title'] = data['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
    
    # Group rare titles
    title_mapping = {
        'Mr': 'Mr',
        'Miss': 'Miss',
        'Mrs': 'Mrs',
        'Master': 'Master',
        'Dr': 'Rare',
        'Rev': 'Rare',
        'Col': 'Rare',
        'Major': 'Rare',
        'Mlle': 'Miss',
        'Countess': 'Rare',
        'Ms': 'Miss',
        'Lady': 'Rare',
        'Jonkheer': 'Rare',
        'Don': 'Rare',
        'Dona': 'Rare',
        'Mme': 'Mrs',
        'Capt': 'Rare',
        'Sir': 'Rare'
    }
    data['Title'] = data['Title'].map(lambda x: title_mapping.get(x, 'Rare'))
    
    # Create family size feature
    data['FamilySize'] = data['SibSp'] + data['Parch'] + 1
    
    # Create is_alone feature
    data['IsAlone'] = (data['FamilySize'] == 1).astype(int)
    
    # Select features for the model
    features = ['Pclass', 'Sex', 'Age', 'Fare', 'Embarked', 'Title', 'FamilySize', 'IsAlone']
    
    if is_training:
        X = data[features]
        y = data['Survived']
        return X, y
    else:
        return data[features]

# Preprocess the training data
X, y = preprocess_data(train_data)

# Define preprocessing pipeline
numeric_features = ['Age', 'Fare', 'FamilySize']
categorical_features = ['Pclass', 'Sex', 'Embarked', 'Title', 'IsAlone']

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Create and train the model
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=1000, random_state=42))
])

# Split the data for validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Evaluate on validation set
y_pred = model.predict(X_val)
y_prob = model.predict_proba(X_val)[:, 1]

print("\nModel Evaluation on Validation Set:")
print(f"Accuracy: {accuracy_score(y_val, y_pred):.4f}")
print(f"Precision: {precision_score(y_val, y_pred):.4f}")
print(f"Recall: {recall_score(y_val, y_pred):.4f}")
print(f"F1 Score: {f1_score(y_val, y_pred):.4f}")
print(f"ROC AUC Score: {roc_auc_score(y_val, y_prob):.4f}")

print("\nConfusion Matrix:")
print(confusion_matrix(y_val, y_pred))

print("\nClassification Report:")
print(classification_report(y_val, y_pred))

# Cross-validation
cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
print(f"\nCross-Validation Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# Plot ROC curve
plt.figure(figsize=(8, 6))
fpr, tpr, _ = roc_curve(y_val, y_prob)
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc_score(y_val, y_prob):.4f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.savefig('roc_curve.png')
plt.close()

# Feature importance
if hasattr(model[-1], 'coef_'):
    # Get feature names after preprocessing
    feature_names = []
    for name, transformer, features in preprocessor.transformers_:
        if name == 'num':
            feature_names.extend(features)
        else:
            # For categorical features, get the one-hot encoded feature names
            encoder = transformer.named_steps['onehot']
            categories = encoder.categories_
            for i, category in enumerate(categories):
                for cat in category:
                    feature_names.append(f"{features[i]}_{cat}")
    
    # Get coefficients
    coefficients = model[-1].coef_[0]
    
    # Create a DataFrame for visualization
    feature_importance = pd.DataFrame({'Feature': feature_names[:len(coefficients)], 'Coefficient': coefficients})
    feature_importance = feature_importance.sort_values('Coefficient', ascending=False)
    
    # Plot feature importance
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Coefficient', y='Feature', data=feature_importance)
    plt.title('Feature Importance (Logistic Regression Coefficients)')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.close()
    
    print("\nFeature Importance:")
    print(feature_importance)

# Train the final model on all training data
model.fit(X, y)

# Save the model
with open('titanic_model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Preprocess and predict on test data
X_test = preprocess_data(test_data, is_training=False)
test_predictions = model.predict(X_test)
test_probabilities = model.predict_proba(X_test)[:, 1]

# Create submission file
submission = pd.DataFrame({
    'PassengerId': test_data['PassengerId'],
    'Survived': test_predictions
})
submission.to_csv('submission.csv', index=False)

print("\nTest predictions saved to 'submission.csv'")