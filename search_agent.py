import math

class SearchAgent:
    k1 = 1.5  # BM25 parameter k1 for tf saturation
    b = 0.75  # BM25 parameter b for document length normalization

    def __init__(self, indexer):
        self.i = indexer

    def query(self, q_str):
        q_str = self.i.clean_text(q_str, True)
        results = []
        for i, d in enumerate(self.i.docs):
            score = self.bm25(d,q_str)
            results.append((i, score))

        results.sort(key= lambda a: a[1], reverse=True)
        if len(results) == 0:
            return None
        else:
            self.display_results(results)

    def bm25(self, doc, query):
        k = 1.5
        b = 0.75
        N = len(self.i.docs)
        avgdl = self.i.corpus_stats['avgdl']
        score = 0
        for term in query:
            tf = doc.count(term)
            df = len(self.i.postings_lists[term])
            dl = len(doc)
            score += math.log((N - df + 0.5) / (df + 0.5) + 1) * (((k + 1) * tf) / (k * ((1 - b) + (b * (dl / avgdl))) + tf))
        return score

    def display_results(self, results):
        # Decode
        for docid, score in results[:5]:  # print top 5 results
            print(f'\nDocID: {docid}')
            print(f'Score: {score}')
            print('Article:')
            print(self.i.raw_ds[docid])