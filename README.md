# 機器學習之分類問題-以鳶尾花資料集為例
- Iris Dataset是由英國的Ronald Fisher於1936年的論文*The use of multiple measurement in taxonomic questions*中作為線性判別分析的一個例子引入的資料集。
- Iris Dataset為基本的機器學習演算法資料，以鳶尾花的花瓣與花萼的長度、寬度預測屬於哪一類的鳶尾花。

## Dataset & Attribute
- 由[UCI](https://archive.ics.uci.edu/ml/datasets/iris)網站取得共150筆資料集，包含三種不同的鳶尾花，每種類別個50筆資料：
  - Setosa（山鳶尾）
  - Versicolor（變色鳶尾）
  - Virginica（維吉尼亞鳶尾）
![](https://i.imgur.com/PGh8wAr.png)

## Algorithms
- [線性區別分析LDA](https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.LinearDiscriminantAnalysis.html?highlight=lineardiscriminantanalysis#sklearn.discriminant_analysis.LinearDiscriminantAnalysis)
- [Logistic迴歸分析](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html?highlight=logisticregression#sklearn.linear_model.LogisticRegression)
- [KNN鄰近法](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html?highlight=kneighborsclassifier#sklearn.neighbors.KNeighborsClassifier)

## Technology
- Python, Anaconda
- Scikit-learn, pandas, Seaborn

---

## Load dataset
- 導入、檢視資料集
    - 將UCI網站的資料集csv傳入，並建立屬性欄位：`sepal-length` `sepal-width` `petal-length` `petal-width` `class`
```python
# Load dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
df = pd.read_csv(url, names=names)

# Dataset
print(df.head(10))
print(df.tail(10))

print(df.describe())
print(df.info())
print(df.shape)
print(df.groupby('class').size())
```

- 資料視覺化
    - 此部分利用Seaborn來實現，Seaborn是基於matplotlib的Python繪圖函式庫。
    - pairplot展示變量的相互關係(線性or非線性)，對角線上為各屬性值的直方圖，而非對角線表示兩個不同屬性之間的關係。
```python
sns.pairplot(df, hue="class", height=2, palette='husl')
```
![](https://i.imgur.com/u3eAsNR.png)

## Building Machine Learning Model

```python
# Split-out validation dataset
array = df.values
X = array[:,0:4]
Y = array[:,4]
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)
scoring = 'accuracy'

# Models
models = []
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('LR', LogisticRegression()))
models.append(('KNN', KNeighborsClassifier()))

# evaluate each model in turn
results = []
names = []
for name, model in models:
    kfold = model_selection.KFold(n_splits=10, random_state=seed)
    cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %.4f (%.4f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)
```

- 線性區別分析LDA
```python
# Make predictions on validation dataset
model = LinearDiscriminantAnalysis()
model.fit(X_train, Y_train)
predictions = model.predict(X_validation)
print('線性區別分析LDA')
print("%.4f" % accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))
print(model)
```
- Logistic迴歸分析
```python
# Make predictions on validation dataset
model = LogisticRegression()
model.fit(X_train, Y_train)
predictions = model.predict(X_validation)
print('Logistic迴歸分析')
print("%.4f" % accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))
print(model)
```
- KNN最近鄰近法
```python
# Make predictions on validation dataset
model = KNeighborsClassifier()
model.fit(X_train, Y_train)
predictions = model.predict(X_validation)
print('KNN最近鄰近法')
print("%.4f" % accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))
print(model)
```


## Model Performance
|           | 線性區別分析LDA | Logistic迴歸分析 | KNN最近鄰近法 |
| --------- | --------------- | ---------------- | ------------- |
| Accuracy  |   0.97          |    0.8           |   0.9         |
| Precision |   0.97          |    0.85          |   0.92        |
| Recall    |   0.97          |    0.83          |   0.91        |
| F1-score  |   0.97          |    0.82          |   0.91        |

綜上，線性區別分析LDA不論在準確率、精確度、召回率、F1分數都優於Logistic回歸分析、KNN最近鄰近法；反之，Logistic迴歸分析在三種演算法中，表現較差。
