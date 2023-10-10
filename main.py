# This is a sample Python script.
import code

from indexer import Indexer
from search_agent import SearchAgent

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    i = Indexer()  # instantiate an indexer
    q = SearchAgent(i)  # document retriever
    code.interact(local=dict(globals(), **locals()))  # interactive shell