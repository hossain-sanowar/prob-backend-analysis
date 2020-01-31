"""
This file contains code to extract selected measurements used in the inTrees
postprocessing framework for tree ensembles [1].
The code is meant to be used on tree ensembles produced by Scikit-Learn.

[1] Deng, H.: Interpreting tree ensembles with inTrees. International Journal
    of Data Science and Analytics 7(4), 277–287 (Jun 2019).
    https://doi.org/10.1007/s41060-018-0144-8
"""

from f109_info import *

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

def analyse_rule_set(rule_set, max_depth=None):
    """
    Calculates the support and confidence for each rule in the rule set.

    Returns a list of tuples `(cond, target, support, confidence)`
    in no guaranteed order.
    """
    analysis = []

    sorted_rules = sorted(rule_set, key=lambda r: 1/len(r[0]))
    supp_div = len(sorted_rules)

    for i in range(len(sorted_rules)):
        rule = sorted_rules[i]
        analysis += [analyse_rule_in_ruleset(rule, sorted_rules[i+1:], max_depth)]
        print("%.02f%%" % (100*(i+1)/supp_div))
    return analysis

def analyse_rule_in_ruleset(rule, rule_set, max_depth=None, file=None, importances=None):
    print("Analysing rule", rule)
    (support_score, confidence) = association_rule_analysis(rule, rule_set, max_depth=max_depth)
    (cond, out) = rule_to_assoc_rule(rule, max_depth=max_depth)
    print("Done with rule", rule, "- Support, confidence:", (support_score, confidence))

    if not (file is None or importances is None):
        w = open(file, "w+")
        pretty_print_assoc_rule((cond, out), importances, w)
        w.write('Support: %d, Confidence: %.2f\n\n' % (support_score, confidence))
        w.close()

    return [cond, out, support_score, confidence]


def pretty_print_assoc_rule(rule, importances, target_file):
    (cond, target) = rule
    for c in sorted(cond, key=lambda x: 1/importances[x[0]]):
        fid, leq = c
        target_file.write(f109_name(fid))
        target_file.write(" (low)" if leq else " (**high**)")
        target_file.write(", importance: %.2f\n" % importances[fid])
    target_file.write("=> %d\n" % target)


def association_rule_analysis(rule, rule_set, max_depth=None):
    """
    Returns a tuple: amount of supporting rules and the confidence.
    """
    support = 0
    confidence = 0

    rule_list = list(rule_set)

    (this_cond, this_target) = rule_to_assoc_rule(rule, max_depth)

    for (other_cond, t) in rule_set:
        if this_cond.issubset(association_rule_cond(other_cond)):
            support += 1
            if t == this_target:
                confidence += 1
    return (support, confidence/support if confidence > 0 else 0)


def rule_to_assoc_rule(rule, max_depth=None):
    """
    Translates a rule into an association rule.
    This returns a tuple `(cond, target)`
    where `cond` is a set of tuples `(feature, leq)`
    with `leq` indicating whether the path used was into the left subtree or
    not.

    By setting a `max_depth=n` value,
    only the first `n` decisions in the rule are considered.
    """
    (cond, target) = rule
    return (association_rule_cond(cond, max_depth), target)


def association_rule_cond(cond, max_depth=None):
    """
    Extracts the condition of a rule as association rule condition.
    For this, the condition is additionally turned into a set.

    The extracted association rule only consists of tuples
    `(feature_id, less_or_equal_threshold)`.
    """
    transformed_cond_list = []
    depth = 0
    for (_, id, _, leq) in cond:
        transformed_cond_list += [(id, leq)]
        depth += 1
        if max_depth != None and max_depth <= depth:
            break
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
