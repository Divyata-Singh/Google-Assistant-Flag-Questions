import pandas as pd
df = pd.read_csv("C:\\Users\\Heather\\Downloads\\labeled_data.csv")
df.drop(['Unnamed: 0','count','hate_speech','offensive_language','neither'],axis=1)
import string
df.tweet = df.tweet.apply(lambda x: x.lower())

df.tweet = df.tweet.apply(lambda x: x.translate(str.maketrans('','','1234567890')))
df.tweet = df.tweet.apply(lambda x: x.translate(str.maketrans('','',string.punctuation)))

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
#
# models = [
#     RandomForestClassifier(n_estimators=200, max_depth=3, random_state=0),
#     LinearSVC(),
#     MultinomialNB(),
#     LogisticRegression(random_state=0),
# ]
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2))#, stop_words='english')
features = tfidf.fit_transform(df.tweet).toarray()
labels = df['class']
# features.shape
# import matplotlib.pyplot as plt
# CV = 5
# cv_df = pd.DataFrame(index=range(CV * len(models)))
# entries = []
# for model in models:
#   model_name = model.__class__.__name__
#   accuracies = cross_val_score(model, features, labels, scoring='accuracy', cv=CV)
#   for fold_idx, accuracy in enumerate(accuracies):
#     entries.append((model_name, fold_idx, accuracy))
# cv_df = pd.DataFrame(entries, columns=['model_name', 'fold_idx', 'accuracy'])
# import seaborn as sns
# sns.boxplot(x='model_name', y='accuracy', data=cv_df)
# sns.stripplot(x='model_name', y='accuracy', data=cv_df,
#               size=8, jitter=True, edgecolor="gray", linewidth=2)
# plt.show()
# cv_df.groupby('model_name').accuracy.mean()
model = LinearSVC()
X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(features, labels, df.index, test_size=0.33, random_state=0)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

import pickle
filename = 'apt.sav'
pickle.dump(model, open(filename, 'wb'))
from sklearn.metrics import confusion_matrix
conf_mat = confusion_matrix(y_test, y_pred)
# fig, ax = plt.subplots(figsize=(10,10))
# sns.heatmap(conf_mat, annot=True, fmt='d')
# plt.ylabel('Actual')
# plt.xlabel('Predicted')
# plt.show()

xt = tfidf.transform(['it is sexy']).toarray()
model.predict(xt)