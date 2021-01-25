import collections
import numpy as np
import gym

from utility.utils import infer_dtype, convert_dtype
import cv2
# stop using GPU
cv2.ocl.setUseOpenCL(False)

# for multi-processing efficiency, we do not return info at every step
EnvOutput = collections.namedtuple('EnvOutput', 'obs reward discount reset')
# Output format of gym
GymOutput = collections.namedtuple('GymOutput', 'obs reward discount')


class FrameStack(gym.Wrapper):
    def __init__(self, env, k):
        super().__init__(env)
        self.k = k
        # self.stack_channel = stack_channel
        self.frames = collections.deque([], maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(shp[:-1] + (shp[-1] * k,)), dtype=env.observation_space.dtype)

    def reset(self):
        obs = self.env.reset()
        for _ in range(self.k):
            self.frames.append(obs)
        return self._get_ob()

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.frames.append(obs)
        return self._get_ob(), reward, done, info

    def _get_ob(self):
        assert len(self.frames) == self.k
        return np.concatenate(self.frames, axis=-1)
            # if self.stack_channel else np.stack(self.frames, axis=0)
            
class ResetSignal(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

        self.observation_space = gym.spaces.Dict({
            'img': self.env.observation_space,
            'reset': gym.spaces.Discrete(2),
        })

    def reset(self):
        obs = self.env.reset()

        return {'img': obs, 'reset': 1}

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        obs = {'img': obs, 'reset': 0}
        return obs, reward, done, info


class RewardHack(gym.Wrapper):
    """ This wrapper should be invoked after EnvStats to avoid inaccurate bookkeeping """
    def __init__(self, env, reward_scale=1, reward_clip=None, **kwargs):
        super().__init__(env)
        self.reward_scale = reward_scale
        self.reward_clip = reward_clip

    def step(self, action, **kwargs):
        obs, reward, done, info = self.env.step(action, **kwargs)
        reward *= self.reward_scale
        if self.reward_clip:
            reward = np.clip(reward, -self.reward_clip, self.reward_clip)
        reward = reward
        return obs, reward, done, info

class EarlyReset(gym.Wrapper):
    def __init__(self, env, step_size):
        super().__init__(env)

        self.episodes = 0
        self.start = 100
        self.cur_epslen = 0
        self.avg_epslen = 0
        self.step_size = step_size
        self.good_eps = True
    
    def reset(self):
        if self.good_eps:
            self.avg_epslen += self.step_size * (self.cur_epslen - self.avg_epslen)
        self.good_eps = True
        return self.env.reset()

    def step(self, action):
        if self.episodes > self.start and cur_epslen > 10 * self.avg_epslen:
            action = self.action_space.sample()
            self.good_eps = False
        obs, rew, done, info = self.env.step(action)
        
        return obs, rew, done, info

def get_wrapper_by_name(env, classname):
    currentenv = env
    while True:
        if classname == currentenv.__class__.__name__:
            return currentenv
        elif hasattr(currentenv, 'env'):
            currentenv = currentenv.env
        else:
            # don't raise error here, only return None
            return None