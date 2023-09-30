import nltk
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = stopwords.words('russian')

description = """Реклама · Навигатор по школьному образованию. Найдите свою альтернативу обычной школе! · Топ-10 Онлайн-школ. С 1 по 11 класс. С зачислением и без. Аттестат гос.образца"""
vectorizer = TfidfVectorizer(stop_words=stop_words)
tfidf_matrix = vectorizer.fit_transform([description])
feature_names = vectorizer.get_feature_names_out()
keywords = [feature_names[i] for i in tfidf_matrix.nonzero()[1]]

print("Ключевые слова:", keywords)
