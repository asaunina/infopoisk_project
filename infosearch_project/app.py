from flask import Flask, render_template, request, session
import pandas as pd
import os
from search import Search  # общий класс поиска

app = Flask(__name__)
app.secret_key = 'my_secret_key'

@app.route('/')
def index():
    """Первая страница: описание и переход на страницу с поиском"""
    return render_template('page_1.html')

@app.route('/search', methods=['GET', 'POST'])
def search_form():
    """Вторая  страница: ввод запроса и выбор вида поиска: bm25, word2vec, fasttext"""
    if request.method == 'POST':

        query = request.form.get('query')
        search_type = request.form.get('search_type')
        top_n = request.form.get('top_n')

        # Store in session to pass to results page
        session['query'] = query
        session['search_type'] = search_type
        session['top_n'] = top_n


        base_dir = os.path.dirname(os.path.abspath(__file__))
        corpus_path = os.path.join(base_dir, 'corpus.json')

        if not os.path.exists(corpus_path):
            corpus_path = 'corpus.json'

        # Индексация и поиск
        search_class = Search(texts_path=corpus_path, method=search_type)
        search_class.fit()
        search_time, results_df = search_class.search(query=query, top_n=int(top_n))

        results = results_df.to_dict('records')
        search_time = round(search_time, 2)
       
        # храним в session, чтобы дальше передать
        
        session['search_time'] = search_time
        session['results'] = results

        # к странице с результатами
        return render_template('page_3.html',
                             query=query,
                             search_type=search_type,
                             top_n=top_n,
                             search_time=search_time,
                             results=results)

    # GET request - страница запроса
    return render_template('page_2.html')

@app.route('/results')
def results():
    """Третья страница: результаты поиска"""

    query = session.get('query', '')
    search_type = session.get('search_type', '')
    search_time = session.get('search_time', '')
    top_n = session.get('top_n', '')
    results = session.get('results', [])

    return render_template('page_3.html',
                         query=query,
                         search_type=search_type,
                         search_time=search_time,
                         top_n=top_n,
                         results=results)

if __name__ == '__main__':
    app.run()