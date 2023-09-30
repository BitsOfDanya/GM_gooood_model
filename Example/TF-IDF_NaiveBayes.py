import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

data = {
    'Категория': ["Бизнес", "Бизнес", "Бизнес", "Бизнес", "Бытовая техника",
                  "Еда и напитки", "Еда и напитки", "Еда и напитки", "Животные", "Электроника", "Электроника", "Электроника"],
    'Тема': ["Бухгалтерские услуги", "Грузоперевозки и транспортные услуги", "Создание и продвижение сайтов", "Юридические услуги",
             "Бытовая техника", "Доставка воды", "Доставка готовых блюд и продуктов", "Кулинария", "Домашние животные", "Компьютерная техника", "Принтеры и МФУ", "Смартфоны и гаджеты"],
    'Описание': ["Описание сайта 1", "Описание сайта 2", "Описание сайта 3"]
}

df = pd.DataFrame(data)

X = df['Описание']
y = df[['Категория', 'Тема']]
le_category = LabelEncoder()
le_theme = LabelEncoder()
y['Категория'] = le_category.fit_transform(y['Категория'])
y['Тема'] = le_theme.fit_transform(y['Тема'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model_category = make_pipeline(TfidfVectorizer(), MultinomialNB()).fit(X_train, y_train['Категория'])
model_theme = make_pipeline(TfidfVectorizer(), MultinomialNB()).fit(X_train, y_train['Тема'])

y_pred_category = model_category.predict(X_test)
y_pred_theme = model_theme.predict(X_test)
print(classification_report(y_test['Категория'], y_pred_category, target_names=le_category.classes_))
print(classification_report(y_test['Тема'], y_pred_theme, target_names=le_theme.classes_))
