from rank_bm25 import BM25Okapi
import numpy as np
import pandas as pd
import time
from preprocess_query import preprocess_query


class InvertedIndexBM25:
    '''
    Класс для построения обратного индекса через bm-25 (библиотечная версия).
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
                

        # наши bm-25 будут здесь
        self.bm25 = None
        # атрибут, который показывает, был ли построен индекс
        self.is_fit = False

    def fit(self):
        '''
        Метод для построения обратного индекса.
        '''
        # строим обратный индекс bm-25 с использованием библиотеки
        self.bm25 = BM25Okapi(self.sentences)
        # обновляем параметр, отвечающий за построение индекса
        self.is_fit = True
        # выводим сообщение
        print(f'''Реализован обратный индекс через BM25.''')
        return self

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
        # получаем список термов из запроса
        query_terms = preprocess_query(query)
        
        # получаем скоры
        scores = self.bm25.get_scores(query_terms)
        
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
        
        # выводим сообщение и возвращаем датафрейм
        print(f'Время поиска: {(end_time - start_time):.4f} секунд.')
        print(f'Результаты поиска по запросу "{query}" (топ {top_n}):')
        return top_n_scores_df