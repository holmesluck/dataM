import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
import numpy as np
import matplotlib.pyplot as plt
train = pd.read_csv("./all/train.csv")
test = pd.read_csv("./all/test.csv")
check = pd.read_csv("./all/gender_submission.csv")


# print all data and set the display settings for the dataframe
# print(train.shape)
# print(test.shape)
pd.set_option('display.max_rows', 891)
pd.set_option('display.max_columns',12)
pd.set_option('display.width',1000)
# print(train.describe())
# print("###################################################")
# print(test.describe())
# print(train.head())
# print("###################################################")
# print(test.head())
# print(train.info())
# print(test.info())


# from the first step we found that there were many missing value in the dataset and
# some categories value in the dataset so we need to do the preprocessing at first to
# solve these problem
# as we know we need to find the relation with the survival percentage for the passengers so
# we could drop the column of the passengers name, id,cabin, ticket values
# and then we could get two new train dataset and test dataset

train.drop(['Ticket','Name','Cabin'], axis=1, inplace= True)
test.drop(['Ticket','Name','Cabin'],axis=1,inplace= True)

# print(train)
# then we need transfer the sex data
def full(x):
    if x == 'male':
      return  1
    else:
      return 0


train['Sex'] = train['Sex'].apply(full)
test['Sex'] = test['Sex'].apply(full)
# print(train)

# use seaborn to do the vitualizatio
# sns.countplot(x='Embarked',data = train)

# plt.show()

# because the s is largest part of the data so we decide to use the S to fillup the missing column
train['Embarked'] = train['Embarked'].fillna('S')
test['Embarked'] = test['Embarked'].fillna('S')
# generate the number of the state S Q C
labels = train['Embarked'].unique().tolist()
labels1 = test['Embarked'].unique().tolist()

# transfer the SQC to the unique number
train['Embarked'] = train['Embarked'].apply(lambda n: labels.index(n))
test['Embarked'] = test['Embarked'].apply(lambda n: labels.index(n))

# fill all the age numbers
sns.set(style='darkgrid', palette='muted', color_codes=True)
# x = train[train['Age'].notnull()]['Age']
# sns.distplot(x)
x = test[test['Fare'].notnull()]['Fare']
sns.distplot(x)

# because the value visualization show the data structure like a normal distribution
# so we select the mean value of the age to fill the NA
train['Age'] = train['Age'].fillna(train['Age'].mean())
test['Age'] = test['Age'].fillna(test['Age'].mean())
# print the number of the null value of the Age in the dataset
a = train['Age'].isnull().sum()

# fill the null value with a mean value
test['Fare'] = test['Fare'].fillna(test['Fare'].mean())

# split the data set
x = train.iloc[:,2:]
y = train['Survived']
test_x = test.iloc[:,1:]
X_train,X_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=22)

# print(X_train.head(5))
# print(X_test.head(5))
# print(y_train.head(5))
# print(y_test.head(5))


# train the model and assessment

clf = DecisionTreeClassifier()
clf.fit(X_train,y_train)
score = clf.score(X_test,y_test)

print(score)

# use the cross validation to validate the model accuracy

result = cross_val_score(clf,x,y,cv=10)
print(result.mean())


# optimization by using the gridsearch() and a stable range
# we could control the depth of the decision tree
# 1st way to select the stable range to test the cross validation and training score
def cv_score(d):
    clf = DecisionTreeClassifier(max_depth=d)
    clf.fit(X_train,y_train)
    t_score = clf.score(X_train,y_train)
    ts_score = clf.score(X_test,y_test)
    return t_score,ts_score

depth = range(2,30)
scores = [cv_score(d) for d in depth]
t_scores = [s[0] for s in scores]
ts_scores = [s[1] for s in scores]

best_score_index = np.argmax(ts_scores)
best_score = ts_scores[best_score_index]
best_param = depth[best_score_index]
print('best param:{0};best score{1}'.format(best_param,best_score))

# get the predict the value for the test dateset
clf = DecisionTreeClassifier(max_depth=best_param)
clf.fit(X_train,y_train)
# print(test_x)
# print("--------------------------")
# print(X_train)
predict_result = clf.predict(test_x)
# input the training result into a required frame
predict = pd.DataFrame(predict_result,columns=['Survived'])
predict.insert(0,'PassengerId',test['PassengerId'])
predict.to_csv('./all/Decision_Tree_ExpResult.csv',index=False)
# plot the pic
plt.figure(figsize=(10,6),dpi=200)
plt.grid()
plt.xlabel("max depth of the decision tree")
plt.ylabel('the best score')
plt.plot(depth,ts_scores,'.g-',label='cross-validation score')
plt.plot(depth,t_scores,'.r-',label='training score')
plt.legend()
plt.show()

# the 2nd way to find the best score value for the model
# we use the gridsearchCV method from the sklearn
def create():
 threshholds = np.linspace(0,0.5,50)
 parm_grid = { 'criterion': ['gini','entropy'],
              'min_impurity_decrease':threshholds,
              'max_depth':range(2,30)
 }
 clf = GridSearchCV(DecisionTreeClassifier(),parm_grid,cv=15)
 clf.fit(x,y)

 print('best param:{0};best score{1}'.format(clf.best_params_,clf.best_score_))
 best_value = clf.best_params_
 best_mark = clf.best_score_
 return best_value , best_mark

# for i in range(30):
#     print('this is the '+str(i)+' iteration')
#     a ,b =create()
#     a['max_depth']