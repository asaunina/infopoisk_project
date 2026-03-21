import json
from bm25 import InvertedIndexBM25
from word2vec import IndexWord2Vec
from fasttext import IndexFasttext


class Search:
    '''
    Класс для реализации поиска и получения документов.
    Получает метод и реализует поиск по нему.
    '''
    def __init__(self, texts_path, method):
        with open(texts_path, 'r', encoding='utf-8') as f:
            self.corpus = json.load(f)
        
        self.method = method

    def fit(self):
        '''
        Инициализирует индекс, соответствующий указанному при инициализации методу.
        '''
        if self.method == 'bm25':
            self.index = InvertedIndexBM25(self.corpus)
        elif self.method == 'word2vec':
            self.index = IndexWord2Vec(self.corpus)
        elif self.method == 'fasttext':
            self.index = IndexFasttext(self.corpus)
        else:
            raise ValueError('Неверно указан метод. Поддерживаются методы bm25, word2vec, fasttext.')
        
        self.index.fit()
    
    def search(self, query, top_n):
        '''
        Реализует поиск по заданному запросу и top-n.
        '''
        return self.index.search(query, top_n)