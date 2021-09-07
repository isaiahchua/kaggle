import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

class NestedCV():
    
    def __init__(self, X, y, model, xtrials=30, p_grid=None): 
        if isinstance(model(), LogisticRegression) and p_grid == None:
            p_grid = [ {'solver': ['newton-cg', 'lbfgs'],
                        'penalty': ['l2'],
                        'C': [100, 10, 1.0, 0.1, 0.01],
                        'max_iter': [2000]},
                       {'solver': ['saga', 'liblinear'],
                        'penalty': ['l1'],
                        'C': [100, 10, 1.0, 0.1, 0.01],
                        'max_iter': [2000]},
                       {'solver': ['saga'],
                        'penalty': ['elasticnet'],
                        'C': [100, 10, 1.0, 0.1, 0.01],
                        'max_iter': [2000]} ]
        if isinstance(model(), KNeighborsClassifier) and p_grid == None:
            p_grid = [ {'n_neighbours': range(1, 21, 2),
                        'metric': ['euclidean', 'manhattan', 'minkowski'],
                        'weights': ['uniform', 'distance']} ]
        if isinstance(model(), GaussianNB) and p_grid == None:
            p_grid = {}
        if isinstance(model(), SVC) and p_grid == None:
            p_grid = [ {'kernel': ['linear', 'poly', 'rbf', 'sigmoid']},
                       {'C': [100, 10, 1.0, 0.1, 0.001]} ]
        if isinstance(model(), SVC) and p_grid == None:
            p_grid = [ {'criterion': ['gini', 'entropy']},
                       {'splitter': ['best', 'random']},
                       {'class_weight': [None, 'balanced']} ]
        if isinstance(model(), RandomForestClassifier) and p_grid == None:
            p_grid = [ {'max_features': ['sqrt', 'log2']},
                       {'n_estimators': [10, 100, 1000]} ]
        if isinstance(model(), XGBClassifier) and p_grid == None:
            p_grid = [ {'learning_rate': [0.001, 0.01, 0.1, 0.3]},
                       {'n_estimators': range(100, 500, 100)},
                       {'subsample': [0.3, 0.4, 0.5, 0.6, 0.7]} ]
        if isinstance(model(), DecisionTreeClassifier) and p_grid == None:
            p_grid = [ {'criterion': ['gini', ['entropy']]},
                       {'splitter': ['best', 'random']}, 
                       {'max_features': ['sqrt', 'log2']} ]
        self.X = X
        self.y = y
        self.model = model()
        self.xtrials = xtrials
        self.fit_summary = None
        self.p_grid = p_grid
        self.best_score = 0
        self.best_std = 0
        self.best_param = dict()
        self.clf = model()

    def validate(self, X=None, y=None):
        if X == None:
            X = self.X
        if y == None:
            y = self.y
        best_score = 0
        best_std = 0
        best_param = dict()
        mean_scores = np.zeros(self.xtrials)
        std_scores = np.zeros(self.xtrials) 
        params = [{} for _ in range(self.xtrials)]
        results = dict()
        for i in range(self.xtrials):
            inner_cv = KFold(n_splits=5, shuffle=True, random_state=i)
            outer_cv = KFold(n_splits=5, shuffle=True, random_state=i)
            clf = GridSearchCV(estimator=self.model, param_grid=self.p_grid, cv=inner_cv)
            clf.fit(X, y)
            params[i] = clf.best_params_
            score = cross_val_score(clf, X=X, y=y, cv=outer_cv) # GridSearchCV object uses the best clf when predicting
            mean_scores[i] = score.mean()
            std_scores[i] = score.std()
            if score.mean() > best_score:
                best_score = score.mean()
                best_std = score.std()
                best_param = clf.best_params_
        results['mean_scores'] = mean_scores
        results['std_scores'] = std_scores
        results['params'] = params
        self.fit_summary = pd.DataFrame(results)
        self.best_param = best_param
        self.best_score = best_score
        self.best_std = best_std

    def fit(self, X=None, y=None):
        if X == None:
            X = self.X
        if y == None:
            y = self.y
        self.clf.set_params(**self.best_param)
        self.clf.fit(X, y)
        
    def predict(self, X):
        return self.clf.predict(X)

    def reset_clf(self):
        self.clf = self.model

if __name__=='__main__':

    NUM_TRIALS = 30

    # import train and test datasets
    X_train = pd.read_csv('train.csv')
    X_test = pd.read_csv('test.csv')

    # probing dataset structure
    print(X_train.dtypes)
    print(X_train.shape)
    print(X_train.head(25))
    print(X_test.head())

    # split dependent variable column from training set
    y_train = X_train.pop('Survived')

    # Look for any nan values in the independent variable columns in the training
    # and test sets
    for i in range(len(X_train.columns.values)):
        col = X_train.columns.values[i]
        has_nan = X_train[col].isnull().values.any()
        num = X_train[col].isnull().sum()
        print(f'{col}: {has_nan}, {num}')
    for i in range(len(X_test.columns.values)):
        col = X_test.columns.values[i]
        has_nan = X_test[col].isnull().values.any()
        num = X_test[col].isnull().sum()
        print(f'{col}: {has_nan}, {num}')
    np.where(X_test['Fare'].isnull())[0]
    print(X_test['Fare'][152])

    # Handle missing data
    from sklearn.impute import SimpleImputer, MissingIndicator
    imputer_age = SimpleImputer(missing_values=np.nan, strategy='mean')
    X_train.iloc[:, 4:5] = imputer_age.fit_transform(X_train.iloc[:, 4:5])
    imputer_embarked = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
    X_train.iloc[:, -1:] = imputer_embarked.fit_transform(X_train.iloc[:, -1:])
    imputer_cabin = MissingIndicator()
    X_train.iloc[:, -2:-1] = imputer_cabin.fit_transform(X_train.iloc[:, -2:-1]).astype(int)
    imputer_fare = SimpleImputer()
    imputer_fare.fit(X_train.iloc[:, -3:-2])
    X_test.iloc[:, 4:5] = imputer_age.transform(X_test.iloc[:, 4:5])
    X_test.iloc[:, -1:] = imputer_embarked.transform(X_test.iloc[:, -1:])
    X_test.iloc[:, -2:-1] = imputer_cabin.transform(X_test.iloc[:, -2:-1]).astype(int)
    X_test.iloc[:, -3:-2] = imputer_fare.transform(X_test.iloc[:, -3:-2])

    # remove columns not used in machine learning 
    X_train = X_train.drop(['PassengerId', 'Name', 'Ticket'], 1)
    X_test_id = X_test.pop('PassengerId')
    X_test = X_test.drop(['Name', 'Ticket'], 1)

    # label encoding categorical data column
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    X_train['Sex'] = le.fit_transform(X_train['Sex'])
    X_test['Sex'] = le.transform(X_test['Sex'])

    # One hot encoding categorical data column
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder
    ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [-1])], remainder='passthrough')
    X_train = np.array(ct.fit_transform(X_train))
    X_test = np.array(ct.transform(X_test))

    # feature scaling
    from sklearn.preprocessing import StandardScaler 
    sc = StandardScaler()
    X_train[:, 5:8] = sc.fit_transform(X_train[:, 5:8]) 
    X_test[:, 5:8] = sc.transform(X_test[:, 5:8]) 

    # Nested Cross Validation with Grid Search
    classifier = NestedCV(X_train, y_train, RandomForestClassifier, NUM_TRIALS)
    classifier.validate()

    # training logistic regression model
    classifier.fit()
    classifier.fit_summary.to_csv('nestedcv_RandomForest.csv', encoding='utf-8', index=False)
    print(f'Best: {classifier.best_score} ({classifier.best_std}) with: {classifier.best_param}')
    
    # model predictions
    y_pred = classifier.predict(X_test)

    # save predictions as .csv file
    predictions = pd.DataFrame(y_pred, columns=['Survived'])
    predictions.insert(0, column='PassengerId', value=X_test_id)
    predictions.to_csv('titanic_pred_RandomForest.csv', encoding='utf-8', index=False)
