"""Native VectorAI filtered search with explicit search breadth and completeness.
Server returns scores/ranking. No Python cosine or vector reranking is used.
This bounded small-collection workaround compensates for candidate-first filtering.
"""
from actian_vectorai import FilterBuilder,has_id,SearchParams
from .storage import ActianStore,COLLECTION,vector_check

class NativeHistoryStore(ActianStore):
    retrieval_mode='Native Actian filtered vector search; full-collection overfetch and hnsw_ef>=4096'
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.client.vde.open_collection(COLLECTION)
        self.search_audit=[]
    def search(self,vector,eligible_ids):
        vector_check(vector);ids=set(map(str,eligible_ids))
        if len(ids)!=len(eligible_ids):raise ValueError('Duplicate eligible IDs')
        if not ids:return []
        collection=getattr(self,'collection',COLLECTION)
        info=self.client.collections.get_info(collection).model_dump(mode='json')
        count=int(info.get('points_count',info.get('vectors_count',0)))
        if count>10000:raise ValueError('Native full-collection workaround not verified above 10000 points')
        limit=max(1,count,len(ids));ef=max(4096,limit)
        filt=FilterBuilder().must(has_id(list(eligible_ids))).build()
        rows=self.client.points.search(collection,vector,filter=filt,limit=limit,
            params=SearchParams(exact=True,hnsw_ef=ef),with_payload=True,timeout=30)
        hits=[r.model_dump(mode='json') for r in rows]
        actual=[str(h['id']) for h in hits]
        self.search_audit.append(dict(eligible_ids=list(eligible_ids),limit=limit,hnsw_ef=ef,exact=True,returned_ids=actual))
        if len(actual)!=len(set(actual)) or set(actual)!=ids:
            raise RuntimeError('Native filtered search incomplete or out of scope; stop without substitute ranking')
        return hits
