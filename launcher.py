from search_engine import engine

def launch():
    search_engine = engine.SearchEngine()
    search_engine.do_indexing('data.nosync/reuters21578/')
    search_engine.answer_query("Donald Trump", 5, 'okapi')


if __name__ == '__main__':
    launch()