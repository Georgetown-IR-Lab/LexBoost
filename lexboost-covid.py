import pandas as pd
import os
import pyterrier as pt

pt.init()
from ir_measures import *
from pyterrier_pisa import PisaIndex
from corpus_graph import CorpusGraph
import pickle
import os.path
from duplicator import MarcoDuplicator
import json

CUTOFF = # Enter Value e.g. 1000

class CG(pt.Transformer):
    def __init__(self, graph, neighbours, ratio):
        self.graph = graph
        self.neighbours = neighbours
        self.ratio = ratio

    def transform(self, inp):
        # print(inp)
        # assert 'query_vec' in inp.columns and 'qid' in inp.columns
        # assert 'docno' in inp.columns
#        inp = inp.set_index(['qid', 'docno'])

        # print(inp.index.values)
        # exit()
        # print(inp)
        # exit()
        dfdict = inp.groupby('qid')[['query', 'docno', 'score', 'rank']].apply(
            lambda x: x.set_index('docno').to_dict(orient='index')).to_dict()
        # dfdict = inp.groupby('qid')[['query', 'docno', 'score', 'rank']].apply(
        #     lambda x: x.set_index('docno').to_dict(orient='index')).to_dict()

        # print(dfdict)
        # exit()
        res = {'qid': [], 'doc_id': [], 'docno': [], 'score': []}

        ctr = 0
        for qid in dfdict.keys():
            for docid in dfdict[qid].keys():
                score = dfdict[qid][docid]['score']

                nscore = 0
                try:
                    for target_did in self.graph.neighbours(docid):
                        # print(target_did)
                        # print(inp.loc[(inp['qid'] == qid) & (inp['docno'] == target_did)])
                        if target_did in dfdict[qid].keys():
                            nscore += int(dfdict[qid][target_did]['score'])
                except:
                    ctr += 1
                nscore /= self.neighbours
                res['qid'].append(qid)
                res['doc_id'].append(docid)
                res['docno'].append(docid)
                res['score'].append(self.ratio * score + (1 - self.ratio) * nscore)

        res = pd.DataFrame(res)
        res = pt.model.add_ranks(res)
        # print(res)
        #        exit()
        print(ctr)
        return res


def getgraph(value):
    graph = CorpusGraph.load('cord19.tct_colbert-v2-hnp.flex/corpusgraph_k128').to_limit_k(value)
    return graph

# ds = pt.get_dataset('irds:msmarco-passage-v2/trec-dl-2022/judged')
ds = pt.get_dataset('irds:cord19/trec-covid')
bm25index = './cord19'
bm25 = PisaIndex(bm25index).bm25(num_results=CUTOFF)


for r in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]:

    print(r)
    cg2 = CG(getgraph(2), 2, r)
    cg4 = CG(getgraph(4), 4, r)
    cg8 = CG(getgraph(8), 8, r)
    cg16 = CG(getgraph(16), 16, r)

    # result = pt.Experiment(
    #     [bm25 >> cg2],
    #     ds.get_topics(),
    #     ds.get_qrels(),
    #     [nDCG @ 10, nDCG @ 1000, MAP, R(rel=2) @ 1000, 'mrt'],
    #     names=["asff"], baseline=0
    # )

    result = pt.Experiment(
        [bm25, bm25 >> cg2, bm25 >> cg4, bm25 >> cg8, bm25 >> cg16],
        ds.get_topics('title'),
        ds.get_qrels(),
        [nDCG @ 10, nDCG @ 100, nDCG @ 1000, MAP, R(rel=2) @ 1000, 'mrt'],
        names=["BM25", "CG(2)", "CG(4)", "CG(8)", "CG(16)"], baseline=0
    )
    results = result.round(4)
    print(results)
    result.to_csv('resultsco/resco_' + str(int(r * 100)) + '.csv')


