from search_engine.engine import SearchEngine

def launch():
    search_engine = SearchEngine()
    search_engine.do_indexing('data.nosync/reuters21578/')
    
    # okapi/cosine scoring
    search_engine.answer_query('Apple product', 2, scoring='okapi')
    search_engine.answer_query('Apple product', 2, scoring='cosine')
    
    # spell-checking
    search_engine.answer_query('Ap*le', 2)
    search_engine.answer_query("Donld Trunp", 5, scoring='okapi')

    # phrase
    search_engine.answer_query("Democratic party", 4, do_phrase=True)
    
    # exact-inexact comparison
    search_engine.answer_query("Apple products", 1, scoring='okapi', do_inexact=False, print_res=False)
    search_engine.answer_query("Apple products", 1, scoring='okapi', do_inexact=True, print_res=False)

    # lm scoring
    search_engine.answer_query("Democratic party", 2, do_inexact=True, scoring='lm')

    # query expansion
    search_engine.answer_query('Democratic party', 2, do_inexact=True, use_expansion=True)
    
    


if __name__ == '__main__':
    launch()