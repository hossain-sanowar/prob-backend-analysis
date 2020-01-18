"""
Contains functions for collecting stats over the features from the trees of
a random forest.
"""
import numpy as np


def node_neg_class_ratio(tree_, node_id):
    # Code adapted from https://scikit-learn.org/stable/auto_examples/tree/plot_unveil_tree_structure.html
    leaf_ids = []

    children_right = tree_.children_right
    children_left = tree_.children_left

    stack = [(node_id, -1)]  # seed is the root node id and its parent depth
    while len(stack) > 0:
        node_id, parent_depth = stack.pop()
        # If we have a test node
        if (children_left[node_id] != children_right[node_id]):
            stack.append((children_left[node_id], parent_depth + 1))
            stack.append((children_right[node_id], parent_depth + 1))
        else:
            leaf_ids.append(node_id)
    neg_classifications = 0
    for id in leaf_ids:
        nprob, pprob = tree_.value[id][0][0]
        if nprob > pprob:
            neg_classifications += 1
    return neg_classifications/len(leaf_ids)


def gather_split_info(tree, feature):
    tree_ = tree.tree_
    argw = np.argwhere(tree_.feature == feature)
    if len(argw) < 1:
        return {} # Feature not used.
    fid = argw[0] # Find index of node for this feature.
    threshold = tree_.threshold[fid] # If value <= threshold then left child
    neg_rate = node_neg_class_ratio(tree_, fid)
    neg_left = node_neg_class_ratio(tree_, tree_.children_left[fid])
    neg_right = node_neg_class_ratio(tree_, tree_.children_right[fid])
    return {'threshold': threshold, 'unknown_rate': neg_rate, 'unknown_rate_left': neg_left, 'unknown_rate_right': neg_right}


def gather_tree_info(tree):
    tree_ = tree.tree_
    info = {}
    info["max_depth"] = tree_.max_depth
    info["splits"] = [gather_split_info(tree, f) for f in range(275)]
    return info
