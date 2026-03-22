import argparse
import json
import sys
import time
from search import Search


def parse_arguments():
    search_parser = argparse.ArgumentParser(
                        description='Находит топ-n релевантных текстов по запросу в корпусе произведений Л.Н.Толстого',
                        epilog='''examples:
                                # BM25
                                python cli.py --query "война и мир" --method bm25 --top-n 10 --corpus corpus.json
                                
                                # Word2Vec
                                python cli.py --query "счастье" --method word2vec --top-n 5 --corpus corpus.json
                                
                                #FastText 
                                python cli.py --query "Всё смешалось в доме Облонских!" --method fasttext --top-n 20 --corpus corpus.json'''
    )

    # добавление аргументов
    search_parser.add_argument(
        '--query', '-q',
        type=str,
        required=True,
        help='Поисковый запрос'
    )

    search_parser.add_argument(
        '--method', '-m',
        type=str,
        required=True,
        help='Метод поиска/индексации: bm25/fasttext/word2vec'
    )
    search_parser.add_argument(
        '--top-n', '-n',
        type=int,
        default=5,
        help='Число возвращаемых запросов'
    )
    search_parser.add_argument(
        '--corpus', '-c',
        type=str,
        default='corpus.json',
        help='Путь до файла с корпусом'
    )

    return search_parser.parse_args()


def main():
    'Функция поиска с cli аргументами'
    args = parse_arguments()

    search_class = Search(texts_path=args.corpus, method=args.method)
    search_class.fit()
    search_time, results_df = search_class.search(query=args.query, top_n=int(args.top_n))

    # вывод результатов 

    for i, row in results_df.iterrows():
        print('='*25)
        print(f'Текст номер {i+1}')
        print(f'Score: {row['score']}')
        print(f'Название книги: {row['book_title']}')
        print(f'Название главы: {row['chapter_name']}')
        print(f'Текст: "{row['sentence_text']}"')

    print('Конец вывода')


if __name__ == "__main__":
    main()
