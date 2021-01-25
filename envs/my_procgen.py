from ray.tune import registry

from envs.procgen_env_wrapper import ProcgenEnvWrapper
from envs import wrappers


def make_procgen_env(config):
    config = config.copy()
    frame_stack = config.pop('frame_stack', 1)
    env = ProcgenEnvWrapper(config)
    if frame_stack > 1:
        env = wrappers.FrameStack(env, frame_stack)
    
    return env

# Register Env in Ray
registry.register_env(
    "my_procgen",  # This should be different from procgen_env_wrapper
    make_procgen_env,
)
