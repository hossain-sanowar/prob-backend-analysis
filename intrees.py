"""
This file contains code to extract selected measurements used in the inTrees
postprocessing framework for tree ensembles [1].
The code is meant to be used on tree ensembles produced by Scikit-Learn.

[1] Deng, H.: Interpreting tree ensembles with inTrees. International Journal
    of Data Science and Analytics 7(4), 277–287 (Jun 2019).
    https://doi.org/10.1007/s41060-018-0144-8
"""

def extract_rules(forest, max_depth=None):
    """
    Extracts the rules of a random forest as a tuple `(condition, target)`.
    While `target` corresponds to the final classification value,
    `conditions` itself is a lists of tuples
    `(node_id, feature_id, threshold, leq_threshold)`,
    where `feature_id` references the feature in question,
    `threshold` the corresponding splitting value,
    and `leq_threshold` is a boolean indicating whether the variable value
    must be less or equal to the threshold to satisfy the rule.

    The `max_depth` parameter can be set to an integer to indicate how many
    conjuncts are to be used maximally in each rule
    (i.e., feature splits counted from the root).

    Returns a dictionary with the trees of forest.estimators_ as keys and
    the corresponding set of rules as tuples.
    """
    all_rules = {} # Dictionary over the included trees.
    for tree in forest.estimators_:
        tree_rules = rule_extract_(tree.tree_, {}, 0, [])
        all_rules[tree] = tree_rules
    return all_rules


def rule_extract_(tree, rule_set, current_node, rule_head):
    """
    Implementation of `ruleExtract(ruleSet, currentNote, C)` from the paper.
    """
    left_id = tree.children_left[current_node]
    right_id = tree.children_right[current_node]
    is_leaf_note = left_id == right_id

    result = set(rule_set) # Make copy.
    if is_leaf_note:
        nprob, pprob = tree.value[current_node][0]
        prediction = 1 if pprob > nprob else 0
        entry = (tuple(rule_head), prediction)
        result.add(entry)
    else:
        feature = tree.feature[current_node]
        threshold = tree.threshold[current_node]
        for (child, leq_bool) in [(left_id, True), (right_id, False)]:
            rule = list(rule_head) + [(current_node, feature, threshold, leq_bool)]
            child_set = rule_extract_(tree, result, child, rule)
            result = result.union(child_set)
    return result


def extract_conditions(forest, max_depth=None):
    all_conds = {}
    for tree in forest.estimators_:
        tree_conds = cond_extract_(tree.tree_, {}, 0, [], max_depth=max_depth, current_depth=0)
        all_conds[tree] = tree_conds
    return all_conds


def cond_extract_(tree, cond_set, current_node, rule_head,
                  max_depth=None, current_depth=0):
    """
    Implementation of
    `condExtract(condSet, currentNode, C, maxDepth, currentDepth)`
    from the paper.
    """
    current_depth += 1

    left_id = tree.children_left[current_node]
    right_id = tree.children_right[current_node]
    is_leaf_node = left_id == right_id

    result = set(cond_set) # Make copy.
    if is_leaf_node or current_depth == max_depth:
        result = result.union({rule_head[-1]}) # Last value of rule is current node's condition.
    else:
        feature = tree.feature[current_node]
        threshold = tree.threshold[current_node]
        for (child, leq_bool) in [(left_id, True), (right_id, False)]:
            rule = rule_head + [(feature, threshold, leq_bool)]
            child_set = cond_extract_(tree, result, child, rule,
                                      max_depth, current_depth)
            result = result.union(child_set)
    return result


def rule_frequency(tree, rule):
    tree_ = tree.tree_
    root_neg, root_pos = tree_.value[0][0]
    root_count = root_neg + root_pos

    (rule_cond, _) = rule
    leaf_id, _, _, _ = rule_cond[-1]
    leaf_neg, leaf_pos = tree_.value[leaf_id][0]
    leaf_count = leaf_neg + leaf_pos

    return leaf_count/root_count


def rule_error(tree, rule):
    tree_ = tree.tree_
    (rule_cond, _) = rule
    leaf_id, _, _, _ = rule_cond[-1]
    leaf_neg, leaf_pos = tree_.value[leaf_id][0]
    leaf_count = leaf_neg + leaf_pos

    error_dividend = leaf_neg if leaf_neg < leaf_pos else leaf_pos

    return error_dividend / leaf_count


def rule_length(tree, rule):
    (rule_cond, _) = rule
    return len(rule_cond)


def association_rule_analysis(rule, rule_set):
    """
    Returns a triple: The support, the confidence, and length of the
    given rule.
    """
    support = 0
    supp_c = 0

    confidence = 0
    conf_c = 0

    (rule_cond, rule_target) = rule
    assoc_rule_cond = association_rule_cond(rule_cond)
    for tree in rule_set:
        for (cond, t) in rule_set[tree]:
            supp_c += 1
            if assoc_rule_cond.issubset(association_rule_cond(cond)):
                support += 1
                conf_c += 1
                if t == rule_target:
                    confidence += 1
    return (support/supp_c, confidence/conf_c, rule_length(None, rule))

def association_rule_cond(cond):
    """
    Extracts the condition of a rule as association rule condition.
    For this, the condition is additionally turned into a set.

    The extracted association rule only consists of tuples
    `(feature_id, less_or_equal_threshold)`.
    """
    transformed_cond_list = []
    for (_, id, _, leq) in cond:
        transformed_cond_list += [(id, leq)]
    return set(transformed_cond_list)


# def prune_rule(tree, rule, threshold=0.05, s=1e-6):
#     (rule_cond, rule_pred) = rule
#     E0 = ??
#     pruned_cond = []
#     for cond in rule_cond[::-1]: # Start with last condition.
#         cond_no_i = ??
#         E_no_i = ??
#         decay = (E_no_i - E0)/max([E0, s])
#
#         if decay < threshold:
#             E0 = E_no_i
#         else:
#             pruned_cond.add(cond)
#     return pruned_cond
