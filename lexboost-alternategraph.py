import pandas as pd
import os
print('hi')
import pyterrier as pt
pt.init()
print('hii')
from ir_measures import *
from pyterrier_pisa import PisaIndex
from corpus_graph import CorpusGraph
import pickle
import os.path
from pyterrier_dr import FlexIndex

CUTOFF = # Enter Value e.g. 1000

class CG(pt.Transformer):
    def __init__(self, graph, neighbours, ratio):
        self.graph = graph
        self.neighbours = neighbours
        self.ratio = ratio

    def transform(self, inp):
        dfdict = inp.groupby('qid')[['query', 'docno', 'score', 'rank']].apply(lambda x: x.set_index('docno').to_dict(orient='index')).to_dict()
        res = {'qid': [], 'doc_id': [], 'score': []}

        for qid in dfdict.keys():
            for docid in dfdict[qid].keys():
                score = dfdict[qid][docid]['score']

                nscore = 0
                for target_did in self.graph.neighbours(docid):
                    if target_did in dfdict[qid].keys():
                        nscore += int(dfdict[qid][target_did]['score'])

                nscore /= self.neighbours
                res['qid'].append(qid)
                res['doc_id'].append(docid)
                res['score'].append(self.ratio*score + (1-self.ratio)*nscore)

        res = pd.DataFrame(res)
        res = pt.model.add_ranks(res)
        #print(res)
#        exit()
        return res


def getgraph(value):
    idx = FlexIndex('msmarco-passage.tasb.flex')
    graph = idx.corpus_graph(16)
    return graph


for set in [19, 20]:
    ds = pt.get_dataset('irds:msmarco-passage/trec-dl-20' + str(set) + '/judged')
    bm25 = PisaIndex.from_dataset('msmarco_passage').bm25(num_results=CUTOFF)

    base = bm25

    cg5 = CG(getgraph(16), 16, 0.5)
    cg6 = CG(getgraph(16), 16, 0.6)
    cg7 = CG(getgraph(16), 16, 0.7)
    cg8 = CG(getgraph(16), 16, 0.8)
    cg9 = CG(getgraph(16), 16, 0.9)

    result = pt.Experiment(
        [base, base>>cg5, base>>cg6, base>>cg7, base>>cg8, base>>cg9],
        ds.get_topics(),
        ds.get_qrels(),
        [nDCG@10, nDCG@100, nDCG@1000, MAP, R(rel=2)@1000, 'mrt'], names=["base", "CG(5)", "CG(6)", "CG(7)", "CG(8)", "CG(9)"], baseline=0
    )
    result = result.round(4)
    result.to_csv('resultsbaselines/' + str(set) + '_' + 'tasb' + '.csv')

    print(result)
    #exit()

