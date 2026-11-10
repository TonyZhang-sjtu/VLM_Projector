"""Evidence-contained relation deduplication; original v1 results stay intact.

Never borrow a side from a separate anatomical mention or a different finding.
A partial fact is redundant only when its SAME finding, assertion and anatomy
entity IDs are contained in a more informative compatible fact.
"""
from . import protocol as v1
from .protocol import label_sets, score_sets

VERSION = 'brain_location_graph_audit_v2_contained_relations'


def extract_graph(graph):
    result = v1.extract_graph(graph)
    result['version'] = VERSION
    facts = result['facts']
    kept, removed = [], []
    for i, fact in enumerate(facts):
        redundant = False
        for j, other in enumerate(facts):
            if i == j or fact['finding_entity_id'] != other['finding_entity_id'] or fact['status'] != other['status']:
                continue
            ids, other_ids = set(fact['anatomy_entity_ids']), set(other['anatomy_entity_ids'])
            compatible = all(fact[k] is None or fact[k] == other[k] for k in ['side', 'site'])
            more_complete = any(fact[k] is None and other[k] is not None for k in ['side', 'site'])
            if ids and ids <= other_ids and compatible and more_complete:
                redundant = True
                break
        (removed if redundant else kept).append(fact)
    result['facts'] = kept
    result['contained_redundant_facts'] = removed
    flags = set(result['flags']) - {'positive_location_without_side', 'positive_location_without_site'}
    for key in ['side', 'site']:
        if any(f['status'] == 'present' and f[key] is None for f in kept):
            flags.add('positive_location_without_' + key)
    result['flags'] = sorted(flags)
    return result
