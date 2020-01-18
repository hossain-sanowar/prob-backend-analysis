def collect_feature_relations(rule_set):
    rels = {}
    for (cond, out) in rule_set.items():
        for i in range(len(cond)):
            _, fid, _, leq = cond[i]
            fid_entry = rels.get(fid, {})
            after = fid_entry.get('after', {})
            inc_feature_counters(after, cond[i+1:len(cond)])
            fid_entry['after'] = after
            before = fid_entry.get('before', {})
            inc_feature_counters(before, cond[0:i])
            fid_entry['before'] = before
            rels[fid] = fid_entry
    return rels


def inc_feature_counters(fdict, cond_list):
    """
    Increases counts in the dictionary depending on the list of conditions.
    Dictionary is changed in-place.
    """
    for _, fid, _, leq in cond_list:
        # Increase the count for the feature only for corresponding leq value.
        leq_counts = fdict.get(fid, {})
        leq_counts[leq] = leq_counts.get(leq 0) + 1
        fdict[fid] = leq_counts
    return fdict
