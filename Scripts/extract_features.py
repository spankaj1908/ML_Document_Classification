from processed_data import training_data
from processed_data import test_data
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer(max_features=10000)

X_train = vectorizer.fit_transform(training_data.data)
X_test = vectorizer.transform(test_data.data)
y_train = training_data.target
y_test = test_data.target

