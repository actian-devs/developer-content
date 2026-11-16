"""Pure fixture-testable policy; reference correctness never enters this module."""
from dataclasses import dataclass
import math
from .contract import fingerprint, assert_runtime_only


@dataclass(frozen=True)
class RoutingPolicy:
    similarity_min: float | None = None
    min_messages: int = 5
    min_pass_rate: float = .8
    max_neighbors: int = 10

    def __post_init__(self):
        if self.min_messages != 5 or self.min_pass_rate != .8 or self.max_neighbors != 10:
            raise ValueError('Starting policy is five distinct messages, 80%, up to ten; version changes explicitly')
        if self.similarity_min is not None and (not math.isfinite(self.similarity_min) or not -1 <= self.similarity_min <= 1):
            raise ValueError('Cosine similarity boundary must be finite within [-1,1] or unset')


def decide(hits, snapshot, query_message_key, policy):
    assert_runtime_only(hits)
    if policy.similarity_min is None:
        return dict(selected_model='cloud', reason='similarity_boundary_unset_no_calibrated_default', neighbors=[], qualifying_messages=0, local_runtime_pass_rate=None)
    evidence = snapshot['evidence']
    if {str(h['id']) for h in hits} != set(evidence):
        return dict(selected_model='cloud',reason='incomplete_retrieval_cannot_establish_failure_inclusive_rate',neighbors=[],qualifying_messages=0,local_runtime_pass_rate=None)
    groups = {}
    for hit in hits:
        aid = str(hit['id']); p = hit['payload']; score = float(hit['score'])
        if aid not in evidence or fingerprint(p) != evidence[aid]:
            raise ValueError('Search evidence differs from the frozen retrieval snapshot')
        if p.get('stage') != 'seed' or p.get('model_kind') != 'local' or p.get('config_id') != snapshot['config_id']:
            raise ValueError('Nonseed/cloud/wrong-configuration evidence cannot enter evaluation routing')
        key = p['message_key']
        if key == query_message_key or not math.isfinite(score):
            continue
        group = groups.setdefault(key, dict(message_key=key, message_ids=set(), score=score, runtime_check_pass=True, attempt_ids=[]))
        group['message_ids'].add(p['message_id']); group['score'] = max(group['score'], score)
        group['runtime_check_pass'] &= p['runtime_check_pass'] is True
        group['attempt_ids'].append(aid)
    neighbors = sorted((g for g in groups.values() if g['score']>=policy.similarity_min), key=lambda g:(-g['score'],g['message_key']))[:10]
    for g in neighbors:
        g['message_ids'] = sorted(g['message_ids']); g['attempt_ids'] = sorted(g['attempt_ids'])
    count = len(neighbors); passing = sum(g['runtime_check_pass'] for g in neighbors)
    rate = passing/count if count else None
    local = count >= 5 and passing * 5 >= count * 4
    return dict(selected_model='local' if local else 'cloud', reason='starting_thresholds_met_unvalidated' if local else ('insufficient_distinct_messages' if count < 5 else 'local_runtime_pass_rate_below_80_percent'),
                neighbors=neighbors, qualifying_messages=count, local_runtime_pass_rate=rate,
                evidence_definition='Each distinct normalized message has one vote; all eligible local attempts for it must pass. Failures are retained, not filtered out.')
