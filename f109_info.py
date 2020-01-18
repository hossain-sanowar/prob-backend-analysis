def f109_name(index):
    "Returns the name of the feature."
    mapping = {
        0: "Number of conjuncts (log2)",
        1: "Max conjunct depth per conjunct",
        2: "number of negations per conjunct",
        3: "max negation depth per conjunct",
        4: "max negation depth per number of negations",
        5: "Logic ops per conjunct",
        6: "Conjunctions per logic ops",
        7: "Disjunctions per logic ops",
        8: "Implications per logic ops",
        9: "Equivalences per logic ops",
        10: "Boolean literals per conjunct",
        11: "Boolean conversions per conjunct",
        12: "Quantifiers per conjunct",
        13: "\\forall ratio of all quantifiers",
        14: "\\exists ratio of all quantifiers",
        15: "average nesting depth of quantifiers",
        16: "Equality per conjunct",
        17: "Inequality per conjunct",
        18: "Number of identifiers per conjunct",
        19: "Integer var ratio of identifiers",
        20: "Boolean var ratio of identifiers",
        21: "Set var ratio of identifiers",
        22: "Relation var ratio of identifiers",
        23: "Function var ratio of identifiers",
        24: "#identifier relations per id",
        25: "#ids with bounded domains ratio of all identifers, symbolic",
        26: "#ids with bounded domains ratio of all identifers, explicit",
        27: "#ids with semi-bounded domains ratio of all identifers, symbolic",
        28: "#ids with semi-bounded domains ratio of all identifers, explicit",
        29: "#ids with unbounded domains ratio of all identifers, symbolic",
        30: "#ids with unbounded domains ratio of all identifers, explicit",
        31: "Arithmetic ops per conjunct",
        32: "Addition ratio of arithmetic ops",
        33: "Multiplication ratio of arithmetic ops",
        34: "Division ratio of arithmetic ops",
        35: "Modulo ratio of arithmetic ops",
        36: "Comparissons ratio of arithmetic ops",
        37: "General sum ratio of arithmetic ops",
        38: "General prod ratio of arithmetic ops",
        39: "`succ` ratio of arithmetic ops",
        40: "`pred` ratio of arithmetic ops",
        41: "Set inclusions per conjunct",
        42: "Set operations per conjunct",
        43: "Set comprehensions per conjunct",
        44: "Set memberships per set inclusion op",
        45: "Negative set memberships per set inclusion op",
        46: "Subsets per set inclusion op",
        47: "Strict subsets per set inclusion op",
        48: "Set unions per set set op",
        49: "Intersections per set set op",
        50: "Set subtractions per set set op",
        51: "General set unions per set set op",
        52: "General intersections per set set op",
        53: "Quantified set unions per set set op",
        54: "Quantified intersections per set set op",
        55: "Powersets per conjunct",
        56: "Nested Powerset ration of powersets",
        57: "Powersets per set op",
        58: "avg. power set nesting depth",
        59: "Relations per conjunct",
        60: "Rel ops per conjunct",
        61: "General relations ratio of all relations",
        62: "Total relations ratio of all relations",
        63: "Surjective relations ratio of all relations",
        64: "Bijective relations ratio of all relations",
        65: "Relational images ratio of rel ops",
        66: "Relational inversions ratio of rel ops",
        67: "Relational overrides ratio of rel ops",
        68: "Direct products ratio of rel ops",
        69: "Parallel products ratio of rel ops",
        70: "Relational domain ratio of rel ops",
        71: "Relational range ratio of rel ops",
        72: "prj1 ratio of rel ops",
        73: "prj2 ratio of rel ops",
        74: "forward composition ratio of rel ops",
        75: "Domain restriction ratio of rel ops",
        76: "Domain subtraction ratio of rel ops",
        77: "Range restriction ratio of rel ops",
        78: "Range subtraction ratio of rel ops",
        79: "Functions per conjunct",
        80: "Function applications per conjunct",
        81: "General, partial function ratio over functions",
        82: "General, total function ratio over functions",
        83: "Injective, partial function ratio over functions",
        84: "Injective, total function ratio over functions",
        85: "Surjective, partial function ratio over functions",
        86: "Surjective, total function ratio over functions",
        87: "Bijective, partial function ratio over functions",
        88: "Bijective, total function ratio over functions",
        89: "Lambda-expression ratio over functions",
        90: "Total amount of sequences per conjunct",
        91: "Seq ops per conjunct",
        92: "Normal sequence ratio of all sequences",
        93: "Injective sequence ratio of all sequences",
        94: "`size` calls ratio per seq op",
        95: "`first` calls ratio per seq op",
        96: "`tail` calls ratio per seq op",
        97: "`last` calls ratio per seq op",
        98: "`front` calls ratio per seq op",
        99: "`reverse` calls ratio per seq op",
        100: "`permutation` calls ratio per seq op",
        101: "`concatenation` calls ratio per seq op",
        102: "front insertions ratio per seq op",
        103: "tail insertions ratio per seq op",
        104: "front restrictions ratio per seq op",
        105: "tail restrictions ratio per seq op",
        106: "general concatenations ratio per seq op",
        107: "number of closures per conjunct",
        108: "number of iterations per conjunct"
    }
    return mapping[index]


def f109_category(index):
    "Returns the category of the indexed feature."
    id = (int(index)+1)

    if id < 12: cat = "Logic"
    elif id < 16: cat = "Quantifiers"
    elif id < 18: cat = "Equality"
    elif id < 31: cat = "Identifiers"
    elif id < 41: cat = "Arithmetic"
    elif id < 59: cat = "Set theory"
    elif id < 79: cat = "Relations"
    elif id < 90: cat = "Functions"
    elif id < 107: cat = "Sequences"
    elif id <= 109: cat = "Closure"
    else: cat = "Unknown"

    return cat


def format_importances(importances):
    """
    Formats a list of tuples (importance, feature index) into the 4-tuple
    (gini importance, feature category, feature id, feature name).
    """
    res = []
    for i, f in importances:
        res.append((i, f275_category(f), f, f275_name(f)))
    return res
