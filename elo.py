
import argparse
import numpy as np

parser = argparse.ArgumentParser(description='Compute elo rating')
parser.add_argument('scores', type=str, required=True)
parser.add_argument('--k', type=int, default=32)
parser.add_argument('--rating', type=int, default=1000)
parser.add_argument('--output', type=str, default='./ratings.npy')
parser.add_argument('--seed', type=int, default=1)
args = parser.parse_args()

np.random.seed(args.seed)

def new_rating(p1_r, p2_r, result):
    p_r = np.array([p1_r, p2_r])
    r = 10 ** (p_r / 400)
    e = r / (r[0] + r[1])
    s = np.array([1, 0]) if result == 0 \
        else np.array([0, 1]) if result == 1 \
        else np.array([0.5, 0.5])
    return np.round(p_r + args.k * (s - e))

def main():
    scores = np.load(args.scores)
    ratings = np.full(scores.shape[0], args.rating, dtype=np.int)
    while np.any(scores):
        p1s, p2s, results = np.nonzero(scores)
        idx = np.random.choice(results.size)
        p1, p2, result = p1s[idx], p2s[idx], results[idx]
        ratings[p1], ratings[p2] = new_rating(ratings[p1], ratings[p2], result)
        scores[p1, p2, result] -= 1
    np.save(args.output, ratings)
    print(np.reshape(ratings, (-1, 1)))

if __name__ == '__main__':
    main()
