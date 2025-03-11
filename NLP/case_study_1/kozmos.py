# Required Libraries

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
from surprise.model_selection import train_test_split
from textblob import Word, TextBlob
from wordcloud import WordCloud
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)
pd.set_option('display.float_format', lambda x: '%.2f' % x)

df = pd.read_excel("miuulpythonProject/pythonProject/NLP/case_study_1/amazon.xlsx")
df.head()

# Normalizing Case Folding

df['Review'] = df['Review'].str.lower()

# Removing Punctuation

df['Review'] = df['Review'].str.replace('[^\w\s]', '', regex=True)

# Removing Numerical Values

df['Review'] = df['Review'] .str.replace('\d', '', regex=True)

# Removing Stopwords

sw = stopwords.words('english')
df['Review'] = df['Review'].apply(lambda x: " ".join(x for x in str(x).split() if x not in sw))
df["Review"]

# Removing Rarewords

drops = pd.Series(' '.join(df['Review']).split()).value_counts()[-1000:]
df['Review'] = df['Review'].apply(lambda x: " ".join(x for x in x.split() if x not in drops))

# Lemmatization

df['Review'] = df['Review'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))
df['Review']

# Text Visualization

# Calculation of Term Frequencies

tf = df["Review"].apply(lambda x: pd.value_counts(x.split(" "))).sum(axis=0).reset_index()
tf.columns = ["words", "tf"]
tf.sort_values("tf", ascending=False)

# Barplot

tf[tf["tf"] > 500].plot.bar(x="words", y="tf", color="green", figsize=(10,5))
plt.show()

# Wordcloud

text = " ".join(i for i in df.Review)

wordcloud = WordCloud(max_font_size=70,
                      max_words=100,
                      background_color="white").generate(text)
plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()

# Sentiment Analysis

import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
nltk.download('vader_lexicon')

sia = SentimentIntensityAnalyzer()

df["Review"][0:10].apply(lambda x: sia.polarity_scores(x))

df["Polarity_Score"] = df["Review"][0:10].apply(lambda x: sia.polarity_scores(x)["compound"])

df["Polarity_Score"]

df["Sentiment_Label"] = df["Review"].apply(lambda x: "pos" if sia.polarity_scores(x)["compound"] > 0 else "neg")

df["Sentiment_Label"]

df["Sentiment_Label"].value_counts()

df.groupby("Sentiment_Label")["Star"].mean()

# By labeling comments with Sentiment Intensity Analyzer, the dependent variable for the comment classification machine learning model was created.

###################################
# Machine Learning Preparation

from sklearn.model_selection import train_test_split

train_x, test_x, train_y, test_y = train_test_split(df["Review"],
                                                   df["Sentiment_Label"],
                                                   random_state=42)


# Creating a TF-IDF vectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer()

# Vectorizing the training data
X_train_tfidf = tfidf.fit_transform(train_x)
X_test_tfidf = tfidf.transform(test_x)


# Modeling (Logistic Regression)

# Model training
log_model = LogisticRegression()
log_model.fit(X_train_tfidf, train_y)

# Model performance
y_pred = log_model.predict(X_test_tfidf)
print("Model Performance:")
print(classification_report(y_pred, test_y))

cross_val_score(log_model, X_test_tfidf, test_y, cv=5).mean()
# 0.8546


# Selecting a random comment and making a prediction
random_review = df["Review"].sample(1).iloc[0]
new_review = tfidf.transform([random_review])
pred = log_model.predict(new_review)

print(f"Review: {random_review} \nPrediction: {pred}")


# Modeling (Random Forest)

rf_model = RandomForestClassifier().fit(X_train_tfidf, train_y)
cross_val_score(rf_model, X_test_tfidf, test_y, cv=5, n_jobs=-1).mean()
# 0.8909
