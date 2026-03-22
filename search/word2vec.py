import re
import numpy as np
import pandas as pd
from razdel import sentenize
import time
from preprocess_query import preprocess_query
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import Word2Vec


class IndexWord2Vec:
    '''
    Класс для построения индекса через word2vec.
    '''

    def __init__(self, corpus):
        # при инициализации указываем корпус (вся информация из файла)
        self.corpus = corpus

        self.sentences = []  # список предложений (леммы)
        self.sentences_info = []  # метаинформация для каждого предложения
        self.sentences_text = []  # оригинальный текст предложения
        
        for book_idx, chapter in enumerate(corpus):
            # достаём всю информацию из корпуса
            book_title = chapter['book_title']
            chapter_name = chapter['chapter_name']
            chapter_lemmas = chapter['chapter_text_sent_lemmas']         
            original_sentences = chapter['chapter_sentences']

            # проходимся по каждому предложению в главе
            for sent_idx, sentence_lemmas in enumerate(chapter_lemmas):
                # добавляем леммы для предложения
                self.sentences.append(sentence_lemmas)
                
                # сохраняем метаинформацию
                self.sentences_info.append({
                    'doc_id': len(self.sentences) - 1,  
                    'book_title': book_title,
                    'chapter_name': chapter_name,
                    'sentence_idx': sent_idx,
                    'book_idx': book_idx,
                    
                })

                # также добавляем полный исходный текст предложений
                self.sentences_text.append(original_sentences[sent_idx])
            
                
        # w2v модель
        self.w2v_model = None
        # эмбеддинги для документов (предложений)
        self.doc_embeddings = []
        # атрибут, который показывает, была ли обучена модель
        self.is_fit = False

    def fit(self):
        '''
        Метод для обучения word2vec и построения эмбеддингов документов.
        '''
        # обучаем модель
        self.w2v_model = Word2Vec(self.sentences)

        # строим эмбеддинги документов: усреднение векторов слов в предложении
        for lemmas_sentence in self.sentences:
            vectors_sentence = []
            for lemma in lemmas_sentence:
                if lemma in self.w2v_model.wv:
                    vectors_sentence.append(self.w2v_model.wv[lemma])
            if vectors_sentence:
                sentence_embedding = np.mean(vectors_sentence, axis=0)  
            else:
                sentence_embedding = np.zeros(self.w2v_model.vector_size)           
            
            self.doc_embeddings.append(sentence_embedding)
        
        # превращаем в np.array
        self.doc_embeddings = np.array(self.doc_embeddings)

        # обновляем параметр, отвечающий за обучение модели
        self.is_fit = True
        # выводим сообщение
        print(f'''Обучена модель Word2Vec.''')
        return self
    
    def query_embedding(self, query):
        '''
        Получаем вектор запроса.
        '''
        query_tokens = preprocess_query(query)
        
        # получаем усреднённый вектор запроса (как и для построения усреднённых векторов предложений)
        query_vectors = []
        for token in query_tokens:
            if token in self.w2v_model.wv:
                query_vectors.append(self.w2v_model.wv[token])
        
        if query_vectors:
            return np.mean(query_vectors, axis=0)
        else:
            return np.zeros(self.w2v_model.vector_size)

    def search(self, query, top_n=5):
        '''
        Метод для реализации поиска.
        На вход принимает текст запроса и топ документов, которые нужно вывести. 
        '''
        # если не был реализован метод fit
        if not self.is_fit:
            raise ValueError("Модель не обучена. Необходимо сначала вызвать метод .fit()")
        
        # засекаем время начала поиска
        start_time = time.time()

        # получаем эмбеддинг запроса
        query_embedding = self.query_embedding(query)
        
        # получаем скоры -- косинусная близость
        scores = cosine_similarity([query_embedding], self.doc_embeddings)[0]
        
        # так же форматируем результаты (топ документов с айди, скором и текстом)
        top_n_scores_ids = np.argsort(scores)[::-1][:top_n]

        # засекаем время конца поиска
        end_time = time.time()
        
        results = []
        for sent_id in top_n_scores_ids:
            results.append({
                'sentence_id': int(sent_id),
                'score': float(scores[sent_id]),
                'book_title': self.sentences_info[sent_id]['book_title'],
                'chapter_name': self.sentences_info[sent_id]['chapter_name'],
                'sentence_text': self.sentences_text[sent_id]
            })
        # сохраняем в датафрейм
        top_n_scores_df = pd.DataFrame(results)
        
        # время поиска
        search_time = end_time - start_time
        
        # выводим сообщение и возвращаем датафрейм
        print(f'Время поиска: {search_time:.4f} секунд.')
        print(f'Результаты поиска по запросу "{query}" (топ {top_n}):')
        
        return (search_time, top_n_scores_df)
