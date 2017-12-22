import pandas as pd
import re
import nltk
from nltk.stem.porter import PorterStemmer

nltk.download('stopwords')
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

def clean_data(text):

    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
    text = re.sub('[\W]+', ' ', text.lower()) + ' '.join(emoticons).replace('-', '')
    return text

#unigram tokenizatio of strings
def tokenizer_standard(text):
    return text.split()

#tokenize strings based on nltk porter stemming
def tokenizer_porter(text):
    porter = PorterStemmer()
    return [porter.stem(word) for word in text.split()]

#vectorize tokens using sci-kits built in tf-idf vectorizer
def vectorize(corpus, tokenizer, sw=None):

    tfidf = TfidfVectorizer(lowercase=False,
                                     tokenizer=tokenizer,
                                     preprocessor=None,
                                     stop_words=sw)
    return tfidf.fit_transform(corpus)

#read in csv data from generated csv
df = pd.DataFrame()
df = pd.read_csv('./movie_data.csv')

#clean csv file
df['review'] = df['review'].apply(clean_data)

#logistic regression on data tokenized with standard tokenizer
X_train = vectorize(df['review'].values, tokenizer_standard)
y_train = df['sentiment'].values

lr = LogisticRegression(random_state = 0)
scores = cross_val_score(lr, X_train, y_train, cv = 5)

print("Standard Tokenizer Accuracy: {} (+/- {})".format(scores.mean(), scores.std()))

#logistic regression on data tokenized with portering tokenizer as well as stopwords
stop = stopwords.words('english')

X_train = vectorize(df['review'].values, tokenizer_porter, stop)
y_train = df['sentiment'].values

lr = LogisticRegression(random_state = 0)
scores = cross_val_score(lr, X_train, y_train, cv = 5)

print("Porter Tokenizer Accuracy: {} (+/- {})".format(scores.mean(), scores.std()))