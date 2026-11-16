"""Non-destructive SDK collection bootstrap and strict compatibility verification."""
from actian_vectorai import VectorAIClient,VectorParams,Distance
from google.protobuf.json_format import MessageToDict

def read_config(endpoint,name,api_key=None):
    # SDK 1.0.2's CollectionInfo.from_proto drops config. Read the same SDK
    # generated Get RPC directly; do not infer compatibility from point counts.
    import grpc
    from actian_vectorai.proto import actian_vectorai_collections_pb2 as pb
    from actian_vectorai.proto.actian_vectorai_collections_service_pb2_grpc import CollectionsStub
    with grpc.insecure_channel(endpoint) as channel:
        metadata=[('api-key',api_key),('authorization','Bearer '+api_key)] if api_key else None
        r=CollectionsStub(channel).Get(pb.GetCollectionInfoRequest(collection_name=name),metadata=metadata,timeout=15)
        return MessageToDict(r.result.config,preserving_proto_field_name=True)

def compatibility(config):
    vectors=config.get('params',{}).get('vectors_config',{})
    p=vectors.get('params')
    if not p or int(p.get('size',0))!=768 or str(p.get('distance','')).lower()!='cosine':
        raise ValueError('Collection must have one unnamed 768-dimensional cosine vector; no existing data was reset')
    return dict(size=768,distance='Cosine')

def ensure_collection(client,endpoint,name,api_key=None,config_reader=read_config):
    created=False
    if not client.collections.exists(name):
        try:
            client.collections.create(name,vectors_config=VectorParams(size=768,distance=Distance.Cosine));created=True
        except Exception:
            if not client.collections.exists(name):raise
    verified=compatibility(config_reader(endpoint,name,api_key))
    client.vde.open_collection(name)
    return dict(collection=name,created=created,compatible=True,**verified)
