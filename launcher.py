from search_engine import engine

def launch():
    search_engine = engine.SearchEngine()
    search_engine.index('data.nosync/reuters21578/')


if __name__ == '__main__':
    launch()