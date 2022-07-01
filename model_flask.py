import pandas as pd
from sklearn.model_selection import train_test_split
import category_encoders as ce
import pickle

data = pd.read_csv("D:\streamlit\data_final_clean.csv")
encoder = ce.OrdinalEncoder(cols=['region','language','package'])
df_cv = encoder.fit_transform(data)
df = pd.DataFrame(df_cv)
cv = df.astype({"region":'float', "language":'float', "package":'float', "timezone":'float', "lasttime":'float',"is_paid":'float'})
X = cv.drop(['is_paid'], axis=1)
y = cv['is_paid']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 42)
X_train = encoder.fit_transform(X_train)
X_test = encoder.transform(X_test)

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(X_train,y_train)
pickle.dump(lr, open("model_lr_cv.pkl","wb"))