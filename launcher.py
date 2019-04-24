from search_engine.engine import SearchEngine

def launch():
    search_engine = SearchEngine()
    search_engine.do_indexing('data.nosync/reuters21578/')
    search_engine.answer_query("stocks", 2, 'okapi')
    # search_engine.answer_query("Donld Trunp", 5, 'okapi')


if __name__ == '__main__':
    launch()