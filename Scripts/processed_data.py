from sklearn.datasets import fetch_20newsgroups
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

#let us define the steps for preprocessing in a function
def preprocess_data(data):
    #to lowercase
    data = data.lower()
    # Remove non-alphanumeric characters
    data = re.sub('[^a-z0-9]', ' ', data)
    # Convert the texts in the data into individual words -> make text tokens
    words = data.split()
    # Remove stop words
    updated_tokens = []
    list_stopwords = stopwords.words('english')
    for word in words:
        if word not in list_stopwords:
            updated_tokens.append(word)
    
    # Stemming updated tokens
    stemmed_tokens = []
    stemmer = PorterStemmer()
    for token in updated_tokens:
        stemmed_tokens.append(stemmer.stem(token))
    
    # Join the words back into a single string
    final_text = ' '.join(stemmed_tokens)
    return final_text

def main():

    #fetch raw data from library
    training_data = fetch_20newsgroups(subset='train', remove=('header','footer','quotes'))
    test_data     = fetch_20newsgroups(subset='test', remove=('header','footer','quotes'))

    #fetch the stopwords from nltk library
    nltk.download('stopwords')

    categories = training_data.target_names

    #Let's preprocess all training and test data using this method
    training_data.data = [preprocess_data(text) for text in training_data.data]
    test_data.data     = [preprocess_data(text) for text in test_data.data]

if __name__ == "__main__":
    main()



