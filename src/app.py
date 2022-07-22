from utils import db_connect
engine = db_connect()

# your code here

# Step 0. Load libraries and modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import statsmodels.formula.api as smf
import seaborn as sns
import pickle

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier  # random forest
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import classification_report
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Step 1. Load the dataset
url = 'https://raw.githubusercontent.com/4GeeksAcademy/random-forest-project-tutorial/main/titanic_train.csv'
df = pd.read_csv(url, index_col=0)

# Get basic info
df.info()

# elimino la columna Cabin
df=df.drop(columns='Cabin')

# imputamos en los faltantes de la edad con la meida y la moda en embarked:
df['Age']=df['Age'].fillna(df['Age'].mean())
df['Embarked']=df['Embarked'].fillna(df['Embarked'].mode()[0])

# Transform if needed
# esa transformación afecta a todo el dataset (antes de dividirlo)
X=df.drop(columns=['Ticket','Name','Survived'])
y=df['Survived']

# Split the dataset so to avoid bias
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=1107)

# Estimación modelo Random Forest:
classif=RandomForestClassifier(random_state=1107)

classif.fit(X_train,y_train)

#show predicted dataset:

y_pred=classif.predict(X_test)

cm = confusion_matrix(y_test, y_pred, labels=classif.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classif.classes_)
disp.plot()

plt.show()

print(classification_report(y_test,y_pred))

# Busqueda de hiperparametros:
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start=200, stop=2000, num=10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
#Criterio
criterion = ['gini', 'entropy']
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               #'max_features': max_features, # Son muy pocas variables por lo cual no vale la pena aplicarlo
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap,
               'criterion': criterion}
print(random_grid)


classif_grid=RandomForestClassifier(random_state=1107)
classif_grid_random=RandomizedSearchCV(estimator=classif_grid,n_iter=100,cv=5,random_state=1107,param_distributions=random_grid)

classif_grid_random.fit(X_train,y_train)

classif_grid_random.best_params_

# Mejor bosque:

best_param = classif_grid_random.best_params_

best_param

# usando la mejor combinación de hiperparámetros, estimo modelo final
best_RF=RandomForestClassifier(**best_param)

best_RF.fit(X_train,y_train)

y_pred_best=best_RF.predict(X_test)

cm = confusion_matrix(y_test, y_pred_best, labels=best_RF.classes_)
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm, display_labels=classif.classes_)
disp.plot()

plt.show()



