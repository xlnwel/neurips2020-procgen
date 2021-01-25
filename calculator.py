import argparse
import numpy as np


EASY_GAME_RANGES = {
    'coinrun': [0, 5, 10],
    'starpilot': [0, 2.5, 64],
    'caveflyer': [0, 3.5, 12],
    'dodgeball': [0, 1.5, 19],
    'fruitbot': [-12, -1.5, 32.4],
    'chaser': [0, .5, 13],
    'miner': [0, 1.5, 13],
    'jumper': [0, 1, 10],
    'leaper': [0, 1.5, 10],
    'maze': [0, 5, 10],
    'bigfish': [0, 1, 40],
    'heist': [0, 3.5, 10],
    'climber': [0, 2, 12.6],
    'plunder': [0, 4.5, 30],
    'ninja': [0, 3.5, 10],
    'bossfight': [0, .5, 13],
    'caterpillar': [0, 8.25, 24],
    'gemjourney': [0, 1.1, 16],
    'hovercraft': [0, .2, 18],
    'safezone': [0, .2, 10]
}


def score2reward(env, score):
    low, blind, high = EASY_GAME_RANGES[env]
    rew = score * (high - blind) + blind
    return rew

def reward2score(env, reward):
    low, blind, high = EASY_GAME_RANGES[env]
    score = (reward - blind) / (high - blind)
    return score


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('env', type=str, nargs='*', default=[])
    parser.add_argument('--score', '-s', type=float, nargs='*', default=[])
    parser.add_argument('--reward', '-r', type=float, nargs='*', default=[])
    parser.add_argument('--weight', '-w', type=float, nargs='*', default=[])

    args = parser.parse_args()
    envs = args.env
    scores = args.score
    rewards = args.reward
    weights = np.array(args.weight or [1 for _ in envs])
    weights = weights / np.sum(weights)
    
    if envs == []:
        envs = ['coinrun', 'bigfish', 'miner', 'starpilot', 'chaser', 'plunder',
            'caterpillar', 'gemjourney', 'hovercraft', 'safezone']
        scores = [0.822, 0.717, 0.858, 0.334, 0.776, 0.924,
            0.986, 0.923, 0.930, 0.558]
        weights = [1/12, 1/12, 1/12, 1/12, 1/12, 1/12,
            1/8, 1/8, 1/8, 1/8]
    assert len(envs) == len(scores) == len(weights), (envs, rewards, weights)

    if scores:
        rs = []
        for e, s in zip(envs, scores):
            r = score2reward(e, s)
            rs.append(r)
            print(f'{e}: Score({s}) = Reward({r:.4g})')
        print(f'Weighted Average Rewards: {np.average(rs, weights=weights)}')
    if rewards:
        ss = []
        for e, r in zip(envs, rewards):
            s = reward2score(e, r)
            ss.append(s)
            print(f'{e}: Reward({r}) = Score({s:.4g})')
        print(f'Weighted Average Scores: {np.average(ss, weights=weights)}')