import pandas as pd
import re
import nltk
from nltk.stem.porter import PorterStemmer

nltk.download('stopwords')
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

def clean_data(text):

    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
    text = re.sub('[\W]+', ' ', text.lower()) + ' '.join(emoticons).replace('-', '')
    return text

def tokenizer_standard(text):
    return text.split()

def tokenizer_porter(text):
    porter = PorterStemmer()
    return [porter.stem(word) for word in text.split()]

def remove_stops(tokens):
    stop = stopwords.words('english')
    return [w for w in tokens if w not in stop]

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

X_train = vectorize(df.loc[:25000, 'review'].values, tokenizer_standard)
y_train = df.loc[:25000, 'sentiment'].values

X_test = vectorize(df.loc[25000:, 'review'].values, tokenizer_standard)
y_test = df.loc[25000:, 'sentiment'].values

lr = LogisticRegression(random_state = 0)
lr.fit(X_train, y_train)


stop = stopwords.words('english')
print("Standard Tokenizer Accuracy: {}".format(lr.score(X_train,y_train)))
X_train = vectorize(df.loc[:25000, 'review'].values, tokenizer_porter,stop)
y_train = df.loc[:25000, 'sentiment'].values

X_test = vectorize(df.loc[25000:, 'review'].values, tokenizer_porter,stop)
y_test = df.loc[25000:, 'sentiment'].values

lr = LogisticRegression(random_state = 0)
lr.fit(X_train, y_train)

print("Porter Tokenizer Accuracy: {}".format(lr.score(X_train,y_train)))