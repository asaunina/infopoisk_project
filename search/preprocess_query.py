import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pymorphy3
import re


nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')
morph = pymorphy3.MorphAnalyzer()
stop_words = set(stopwords.words('russian'))


def preprocess_query(query):
    '''
    Функция для предобработки запроса.
    На вход получает запрос (строку).
    На выходе -- леммы (постобработанный запрос).
    '''
    # предобработка та же, что и использовалась при создании корпуса
    query = query.lower()
    query = re.sub(r'[^а-яёa-z\s]', ' ', query)
    query = re.sub(r'\s+', ' ', query).strip()
    tokens = word_tokenize(query, language='russian')

    filtered_tokens = [t for t in tokens if t not in stop_words]

    lemmas = []
    for token in filtered_tokens:
        try:
            lemma = morph.parse(token)[0].normal_form
            lemmas.append(lemma)
        except:
            continue

    return lemmas