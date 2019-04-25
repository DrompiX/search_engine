from search_engine.engine import SearchEngine

def launch():
    search_engine = SearchEngine()
    search_engine.do_indexing('data.nosync/reuters21578/')
    search_engine.answer_query("political analysts", 4, do_phrase=True)
    # search_engine.answer_query("politics", 2, do_inexact=True, use_expansion=True)
    # search_engine.answer_query("politics", 10, do_inexact=True, scoring='lm')
    # search_engine.answer_query("government", 200, do_inexact=False, scoring='cosine')
    # search_engine.answer_query("government", 200, do_inexact=True, scoring='cosine')
    # search_engine.answer_query("Donld Trunp", 5, 'okapi')


if __name__ == '__main__':
    launch()