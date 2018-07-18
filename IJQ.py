import pandas as pd

a = pd.read_csv("./all/gender_submission.csv")
b = pd.read_csv("./all/gender_submission.csv")
pd.set_option("display.max_rows",1000)
pd.set_option("display.width",1000)

print(a.head(5))
print("----------------------------")
print(b.head(5))


predict = pd.DataFrame(a,columns=['Survived'])
predict.insert(0,'PassengerId',b['PassengerId'])
print(predict[0:1])
