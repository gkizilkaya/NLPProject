##################################################
# Introduction to Text Mining and Natural Language Processing
##################################################

##################################################
# Sentiment Analysis and Sentiment Modeling for Amazon Reviews
##################################################

# 1. Text Preprocessing
# 2. Text Visualization
# 3. Sentiment Analysis
# 4. Feature Engineering
# 5. Sentiment Modeling

# !pip install nltk
# !pip install textblob
# !pip install wordcloud


from warnings import filterwarnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, GridSearchCV, cross_validate
from sklearn.preprocessing import LabelEncoder
from textblob import Word, TextBlob
from wordcloud import WordCloud

filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)
pd.set_option('display.float_format', lambda x: '%.2f' % x)

##################################################
# 1. Text Preprocessing
##################################################

df = pd.read_csv("miuulpythonProject/pythonProject/NLP/course_materials/datasets/amazon_reviews.csv", sep=",")
df.head()

###############################
# Normalizing Case Folding
###############################

df['reviewText'] = df['reviewText'].str.lower()

###############################
# Punctuations
###############################

df['reviewText'] = df['reviewText'].str.replace('[^\w\s]', '', regex=True) # regex=True argümanı ChatGPT önerisi

# regular expression

###############################
# Numbers
###############################

df['reviewText'] = df['reviewText'].str.replace('\d', '', regex=True) # regex=True argümanı ChatGPT önerisi

# df['Review'] = df['Review'].str.replace(r'\d+', '', regex=True) (1'den fazla basamaklı sayılar için bu yöntemi önerdi ChatGPT)

###############################
# Stopwords
###############################
import nltk
nltk.download('stopwords')

sw = stopwords.words('english')

df['reviewText'] = df['reviewText'].apply(lambda x: " ".join(x for x in str(x).split() if x not in sw))


###############################
# Rarewords
###############################

temp_df = pd.Series(' '.join(df['reviewText']).split()).value_counts()

drops = temp_df[temp_df <= 1]

df['reviewText'] = df['reviewText'].apply(lambda x: " ".join(x for x in x.split() if x not in drops))

"""
drops = set(temp_df[temp_df <= 1000].index)
drops serisinin index’leri kelimeler, değerleri ise frekanslar olduğundan, .apply() fonksiyonunda karşılaştırma yaparken x not in drops ifadesi doğru çalışmayabilir.
"""


###############################
# Tokenization
###############################

nltk.download("punkt")

# python -m textblob.download_corpora (chatgpt önerisi, terminalde çalıştırdım)

df["reviewText"].apply(lambda x: TextBlob(x).words).head()


###############################
# Lemmatization
###############################

nltk.download('wordnet')

df['reviewText'] = df['reviewText'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))


##################################################
# 2. Text Visualization
##################################################


###############################
# Terim Frekanslarının Hesaplanması
###############################

tf = df["reviewText"].apply(lambda x: pd.value_counts(x.split(" "))).sum(axis=0).reset_index()

tf.columns = ["words", "tf"]

tf.sort_values("tf", ascending=False)

###############################
# Barplot
###############################

tf[tf["tf"] > 500].plot.bar(x="words", y="tf")
plt.show()

###############################
# Wordcloud
###############################

text = " ".join(i for i in df.reviewText) # tüm kelimeleri birleştirdik

wordcloud = WordCloud().generate(text) # text dosyasından wordcloud'u oluştur
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()

# çeşitli ayarlamalarla bi wordcloud yapalım şimdi:

wordcloud = WordCloud(max_font_size=50,
                      max_words=100,
                      background_color="white").generate(text)
plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()

wordcloud.to_file("wordcloud.png") # çıktı görselini kaydedelim


###############################
# Şablonlara Göre Wordcloud
###############################

tr_mask = np.array(Image.open("miuulpythonProject/pythonProject/NLP/course_materials/tr.png"))

wc = WordCloud(background_color="white",
               max_words=1000,
               mask=tr_mask,
               contour_width=3,
               contour_color="firebrick")

wc.generate(text)
plt.figure(figsize=[10, 10])
plt.imshow(wc, interpolation="bilinear")
plt.axis("off")
plt.show()


##################################################
# 3. Sentiment Analysis
##################################################

df["reviewText"].head()

nltk.download('vader_lexicon')

sia = SentimentIntensityAnalyzer()

sia.polarity_scores("The film was awesome") # polarity score pozitif negatif skoru
# {'neg': 0.0, 'neu': 0.423, 'pos': 0.577, 'compound': 0.6249}
# compound skoru önemli bizim için, -1 ve +1 arasında olur, 0'dan büyük olması pozitif olduğunu gösterir.

sia.polarity_scores("I liked this music but it is not good as the other one")
# {'neg': 0.207, 'neu': 0.666, 'pos': 0.127, 'compound': -0.298}

sia.polarity_scores("This music is nice, but the other one is better")
# {'neg': 0.0, 'neu': 0.582, 'pos': 0.418, 'compound': 0.6956} # ozy cümlesi:)


df["reviewText"][0:10].apply(lambda x: sia.polarity_scores(x))

df["reviewText"][0:10].apply(lambda x: sia.polarity_scores(x)["compound"])

df["polarity_score"] = df["reviewText"].apply(lambda x: sia.polarity_scores(x)["compound"])
df.head()

# burada şöyle bir çalışma yapılabilir ödev olarak: polarity sckoru 0.40'tan büyük overall 3'ten küçük olan yorumlar mesela, tezat bi durum bulunur.


###############################
# 4. Feature Engineering
###############################

df["reviewText"][0:10].apply(lambda x: "pos" if sia.polarity_scores(x)["compound"] > 0 else "neg")

df["sentiment_label"] = df["reviewText"].apply(lambda x: "pos" if sia.polarity_scores(x)["compound"] > 0 else "neg")

df["sentiment_label"].value_counts()

df.groupby("sentiment_label")["overall"].mean()

df["sentiment_label"] = LabelEncoder().fit_transform(df["sentiment_label"])

y = df["sentiment_label"]
X = df["reviewText"]

###############################
# Count Vectors
###############################

# Count Vectors: frekans temsiller
# TF-IDF Vectors: normalize edilmiş frekans temsiller
# Word Embeddings (Word2Vec, GloVe, BERT vs)


# words
# kelimelerin nümerik temsilleri

# characters
# karakterlerin numerik temsilleri

# ngram
a = """Bu örneği anlaşılabilmesi için daha uzun bir metin üzerinden göstereceğim.
N-gram'lar birlikte kullanılan kelimelerin kombinasyolarını gösterir ve feature üretmek için kullanılır"""

TextBlob(a).ngrams(3)

###############################
# Count Vectors
###############################

from sklearn.feature_extraction.text import CountVectorizer

corpus = ['This is the first document.',
          'This document is the second document.',
          'And this is the third one.',
          'Is this the first document?']

# word frekans
vectorizer = CountVectorizer()
X_c = vectorizer.fit_transform(corpus)
vectorizer.get_feature_names_out() # eşsiz kelimeleri getiriyorum
# 'and', 'document', 'first', 'is', 'one', 'second', 'the', 'third', 'this']
X_c.toarray() # yukarıdaki eşsiz kelimeler 1. cümlede var mı yok mu 2. cümlede var mı yok mu...
# array([[0, 1, 1, 1, 0, 0, 1, 0, 1],
#        [0, 2, 0, 1, 0, 1, 1, 0, 1],
#        [1, 0, 0, 1, 1, 0, 1, 1, 1],
#        [0, 1, 1, 1, 0, 0, 1, 0, 1]])


# n-gram frekans (kelime öbekleri)
vectorizer2 = CountVectorizer(analyzer='word', ngram_range=(2, 2)) # 2'li ngram, yani 2'li kelime öbekleri olsun diyorum
X_n = vectorizer2.fit_transform(corpus)
vectorizer2.get_feature_names_out()
X_n.toarray()


vectorizer = CountVectorizer()
X_count = vectorizer.fit_transform(X)

vectorizer.get_feature_names_out()[10:15]
X_count.toarray()[10:15]


###############################
# TF-IDF
###############################

from sklearn.feature_extraction.text import TfidfVectorizer
tf_idf_word_vectorizer = TfidfVectorizer() # tfidfvectorizer ön tanımlı değeri word
X_tf_idf_word = tf_idf_word_vectorizer.fit_transform(X)


tf_idf_ngram_vectorizer = TfidfVectorizer(ngram_range=(2, 3))
X_tf_idf_ngram = tf_idf_ngram_vectorizer.fit_transform(X)


###############################
# 5. Sentiment Modeling
###############################

# 1. Text Preprocessing
# 2. Text Visualization
# 3. Sentiment Analysis
# 4. Feature Engineering
# 5. Sentiment Modeling

###############################
# Logistic Regression
###############################

log_model = LogisticRegression().fit(X_tf_idf_word, y)

cross_val_score(log_model,
                X_tf_idf_word,
                y,
                scoring="accuracy",
                cv=5).mean()


new_review = pd.Series("this product is great")
new_review = pd.Series("look at that shit very bad")
new_review = pd.Series("it was good but I am sure that it fits me")

new_review = TfidfVectorizer().fit(X).transform(new_review)

log_model.predict(new_review)

random_review = pd.Series(df["reviewText"].sample(1).values)
# random_review: working year camera crashed failed ive shot hu...

new_review = TfidfVectorizer().fit(X).transform(random_review)

log_model.predict(new_review)


###############################
# Random Forests
###############################

# Count Vectors
rf_model = RandomForestClassifier().fit(X_count, y)
cross_val_score(rf_model, X_count, y, cv=5, n_jobs=-1).mean()
# 0.83

# TF-IDF Word-Level
rf_model = RandomForestClassifier().fit(X_tf_idf_word, y)
cross_val_score(rf_model, X_tf_idf_word, y, cv=5, n_jobs=-1).mean()
# 0.82

# TF-IDF N-GRAM
rf_model = RandomForestClassifier().fit(X_tf_idf_ngram, y)
cross_val_score(rf_model, X_tf_idf_ngram, y, cv=5, n_jobs=-1).mean()
# 0.78

###############################
# Hiperparametre Optimizasyonu
###############################

rf_model = RandomForestClassifier(random_state=17)

rf_params = {"max_depth": [8, None],
             "max_features": [7, "auto"],
             "min_samples_split": [2, 5, 8],
             "n_estimators": [100, 200]}

rf_best_grid = GridSearchCV(rf_model,
                            rf_params,
                            cv=5,
                            n_jobs=-1,
                            verbose=1).fit(X_count, y)

rf_best_grid.best_params_
# {'max_depth': None,
#  'max_features': 7,
#  'min_samples_split': 2,
#  'n_estimators': 100}

rf_final = rf_model.set_params(**rf_best_grid.best_params_, random_state=17).fit(X_count, y)


cross_val_score(rf_final, X_count, y, cv=5, n_jobs=-1).mean()
# 0.81
