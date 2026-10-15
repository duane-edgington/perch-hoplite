#
import csv

def load_scores(path):
    scores = {}
    for row in csv.DictReader(open(path)):
        key = (row['filename'], row['window_start'])
        scores[key] = float(row['logits'])
    return scores

ae0e = load_scores('/mnt/PAM_Analysis/perch-hoplite/results/MARS_20231101_20231130_v10_orcaval.csv')


# load test file
test = load_scores('/mnt/PAM_Analysis/perch-hoplite/results/test_nov2023_0626_v10_orcaval.csv')


common = set(ae0e.keys()) & set(test.keys())
diffs = [abs(ae0e[k] - test[k]) for k in common]

print(f"ae0e detections: {len(ae0e)}")
print(f"0626 detections: {len(test)}")
print(f"Common windows:  {len(common)}")
print(f"Max diff:        {max(diffs):.6f}")
print(f"Mean diff:       {sum(diffs)/len(diffs):.6f}")
print(f"Zero diff:       {sum(1 for d in diffs if d == 0)}/{len(diffs)}")
#

#0626 detections: 975
#Common windows:  975
#Max diff:        0.002188
#Mean diff:       0.000412
#Zero diff:       11/975


