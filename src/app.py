# Aca colocamos el codigo utilizado en el explore.ipynb

import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

df = pd.read_csv('https://raw.githubusercontent.com/4GeeksAcademy/naive-bayes-project-tutorial/main/playstore_reviews_dataset.csv')

#Borramos la columna de nombre package_name

df =df.drop(columns=['package_name'])

df['review']=df['review'].str.lower()
df['review']=df['review'].str.strip()

# Seleccionamos la X y la y para nuestro trabajo

X=df['review']
y=df['polarity']

vectorizer = CountVectorizer(stop_words='english')
X = vectorizer.fit_transform(X)
vectorizer.get_feature_names_out()

X = X.toarray()

# Aca cambio y toma un test y train 0.35 y 0.65
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state=123)

clf = GaussianNB()
clf.fit(X_train, y_train)

y_train_pred = clf.predict(X_train)
y_test_pred = clf.predict(X_test)

target_names = ['bad', 'good']
print(classification_report(y_train, y_train_pred, target_names=target_names))

print(classification_report(y_test, y_test_pred, target_names=target_names))

#No resultó muy bueno - Se ve que tiene bastante  problemas de overfit

clf_multinomial = MultinomialNB()
clf_multinomial.fit(X_train, y_train)

print(classification_report(y_train, y_train_pred, target_names=target_names))

print(classification_report(y_test, y_test_pred, target_names=target_names))

Z = vectorizer.transform(['I like this app'])

#Modelo GaussianNB

print("Modelo GaussianNB",clf.predict(Z.toarray()))

# Modelo Multinomial

print("Modelo Multinomial",clf_multinomial.predict(Z.toarray()))

pd.set_option('display.max_colwidth', None)
df.tail(10)

Z2 = vectorizer.transform(['I do not like this app'])

# Modelo GaussianNB
print(f"Gauss: {clf.predict(Z2.toarray())}")


# Modelo Multinomial
print(f"Multi: {clf_multinomial.predict(Z2.toarray())}")

pickle.dump(clf_multinomial, open('../models/multinomial.pkl', 'wb'))

print("El programa ha finalizado, se ha grabado el archivo")
print("En la carpeta models")




