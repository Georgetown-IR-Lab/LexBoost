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
#from pyterrier_dr import FlexIndex
from pyterrier_dr import FlexIndex, TasB, TctColBert

CUTOFF = # Enter Value e.g. 1000

class CG(pt.Transformer):
    def __init__(self, graph, neighbours, ratio):
        self.graph = graph
        self.neighbours = neighbours
        self.ratio = ratio

    def transform(self, inp):
        dfdict = inp.groupby('qid')[['query', 'docno', 'score', 'rank']].apply(lambda x: x.set_index('docno').to_dict(orient='index')).to_dict()
        res = {'qid': [], 'doc_id': [], 'docno': [], 'score': [], 'query': []}

        for qid in dfdict.keys():

            for docid in dfdict[qid].keys():
                score = dfdict[qid][docid]['score']
                query = dfdict[qid][docid]['query']
                nscore = 0
                for target_did in self.graph.neighbours(docid):
                    if target_did in dfdict[qid].keys():
                        nscore += int(dfdict[qid][target_did]['score'])

                nscore /= self.neighbours
                res['qid'].append(qid)
                res['doc_id'].append(docid)
                res['docno'].append(docid)

                res['score'].append(self.ratio*score + (1-self.ratio)*nscore)
                res['query'].append(query)
        res = pd.DataFrame(res)

        res = res.sort_values(['score'], ascending=False).groupby('qid').head(1000)
        res2 = res.copy()
        res2 = pt.model.add_ranks(res2)

        #print(res2)
#        exit()
        return res2



def getgraph(method):
    if method == 'hnp':
        graph = CorpusGraph.from_dataset('msmarco_passage', 'corpusgraph_tcthnp_k16').to_limit_k(16)
    elif method == 'tas':
        idx = FlexIndex('msmarco-passage.tasb.flex')
        graph = idx.corpus_graph(16)
    return graph

for set in [19, 20]:
    for method in ['tas']:
        for gmethod in ['hnp']:

            ds = pt.get_dataset('irds:msmarco-passage/trec-dl-20' + str(set) + '/judged')

            if method == 'hnp':
                model = TctColBert('castorini/tct_colbert-v2-hnp-msmarco')
                idx = FlexIndex('msmarco-passage.hnp.flex')

            elif method == 'tas':
                model = TasB.dot()
                idx = FlexIndex('msmarco-passage.tasb.flex')

            base = PisaIndex.from_dataset('msmarco_passage').bm25(num_results=CUTOFF)
            #rr = model.query_encoder() >> idx.np_retriever(num_results=1000)
            #rr =  model.query_encoder() >> idx.faiss_flat_retriever(qbatch=1)
            rr = model.query_encoder() >> idx.scorer(num_results=1000)

            cg5 = CG(getgraph(gmethod), 16, 0.5)
            cg6 = CG(getgraph(gmethod), 16, 0.6)
            cg7 = CG(getgraph(gmethod), 16, 0.7)
            cg8 = CG(getgraph(gmethod), 16, 0.8)
            cg9 = CG(getgraph(gmethod), 16, 0.9)

            #print(base(ds.get_topics()))
            #exit()
            result = pt.Experiment(
                [base %1000 >> rr , base>>cg5 >> rr, base>>cg6 >> rr, base>>cg7 >> rr, base>>cg8 >> rr, base>>cg9 >> rr],
                ds.get_topics(),
                ds.get_qrels(),
                [nDCG@10, nDCG@100, nDCG@1000, MAP, R(rel=2)@1000, 'mrt'], names=["base", "CG(5)", "CG(6)", "CG(7)", "CG(8)", "CG(9)"], baseline=0
            )
            result = result.round(4)
            result.to_csv('resultrerank/' + str(set) + '_' + method + '_' + gmethod + '.csv')

            print(result)
            #exit()

