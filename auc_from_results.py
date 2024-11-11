import numpy as np
import sklearn.metrics
import sys, os, csv

# if len(sys.argv) < 2:
#     print("Usage: auc_from_results.py [results.csv]")
#     sys.exit(1)

def calc_auc(no_wm_scores, wm_scores):
    assert no_wm_scores.shape == wm_scores.shape
    labels = np.array([0] * len(no_wm_scores) + [1] * len(wm_scores))
    scores = np.concatenate((no_wm_scores, wm_scores))
    return sklearn.metrics.roc_auc_score(labels, scores)

def delim_print(l, delim="\t"):
    print(delim.join(str(x) for x in l))

no_wm_suffix = "_no_wm"
wm_suffix = "_wm"

def extract_auc_scores(filename):
    # Extract the test case names from the header
    test_cases = []
    with open(filename, newline='') as fp:
        csv_reader = csv.reader(fp)
        # Skip the config line
        next(csv_reader)
        header = next(csv_reader)
        assert header[0] == "i"
        for i in range(1, len(header), 2):
            assert header[i].endswith(no_wm_suffix)
            assert header[i+1].endswith(wm_suffix)
            assert header[i][:-len(no_wm_suffix)] == header[i+1][:-len(wm_suffix)]
            test_cases.append(header[i][:-len(no_wm_suffix)])

    data = np.genfromtxt(filename, delimiter=',', skip_header=2)
    assert data.shape[1] == len(test_cases) * 2 + 1
    auc_scores = [ calc_auc(data[:, 2*i + 1], data[:, 2*i + 2]) for i in range(len(test_cases)) ]
    return data.shape[0], test_cases, auc_scores

files = sys.argv[1:]

test_cases = None
for f in files:
    n, tests, auc_scores = extract_auc_scores(f)
    if test_cases is None:
        delim_print(["method", "n"] + tests)
        test_cases = tests
    if tests != test_cases:
        print("WARNING! test cases changed")
        print("was:", test_cases)
        print("now:", tests)
    delim_print([f, n] + auc_scores)
