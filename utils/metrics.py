import numpy as np

def quadratic_weighted_kappa(rater_a, rater_b, min_rating=None, max_rating=None):
    rater_a = np.array(rater_a, dtype=int)
    rater_b = np.array(rater_b, dtype=int)
    assert(len(rater_a) == len(rater_b))
    if min_rating is None:
        min_rating = min(min(rater_a), min(rater_b))
    if max_rating is None:
        max_rating = max(max(rater_a), max(rater_b))
    conf_mat = confusion_matrix_(rater_a, rater_b, min_rating, max_rating)
    num_scored_items = float(conf_mat.sum())
    num_ratings = float(len(conf_mat))
    hist_rater_a = histogram(rater_a, min_rating, max_rating)
    hist_rater_b = histogram(rater_b, min_rating, max_rating)
    numerator = 0.0
    denominator = 0.0
    for i in range(len(conf_mat)):
        for j in range(len(conf_mat)):
            expected_count = (hist_rater_a[i] * hist_rater_b[j]
                              / num_scored_items)
            d = pow(i - j, 2.0) / pow(max_rating - 1, 2.0)
            numerator += d * conf_mat[i][j] / num_scored_items
            denominator += d * expected_count / num_scored_items
    return 1.0 - numerator / denominator

def confusion_matrix_(rater_a, rater_b, min_rating=None, max_rating=None):
    assert(len(rater_a) == len(rater_b))
    if min_rating is None:
        min_rating = min(min(rater_a), min(rater_b))
    if max_rating is None:
        max_rating = max(max(rater_a), max(rater_b))
    num_ratings = max_rating - min_rating + 1
    conf_mat = [[0 for i in range(int(num_ratings))]
                for j in range(int(num_ratings))]
    for a, b in zip(rater_a, rater_b):
        conf_mat[a - min_rating][b - min_rating] += 1
    return np.array(conf_mat)

def histogram(ratings, min_rating=None, max_rating=None):
    if min_rating is None:
        min_rating = min(ratings)
    if max_rating is None:
        max_rating = max(ratings)
    hist_ratings = [0 for x in range(min_rating, max_rating + 1)]
    for r in ratings:
        hist_ratings[r - min_rating] += 1
    return hist_ratings