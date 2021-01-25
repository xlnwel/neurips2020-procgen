import copy
import logging
import collections

import ray
from ray.rllib.utils.deprecation import deprecation_warning, DEPRECATED_VALUE
from ray.rllib.utils.actors import create_colocated
from ray.rllib.execution.common import STEPS_TRAINED_COUNTER, \
    SampleBatchType, _get_shared_metrics, _get_global_vars
from ray.rllib.evaluation.worker_set import WorkerSet
from ray.rllib.execution.rollout_ops import ParallelRollouts
from ray.rllib.execution.concurrency_ops import Concurrently, Enqueue, Dequeue
from ray.rllib.execution.replay_ops import StoreToReplayBuffer, Replay
from ray.rllib.execution.train_ops import UpdateTargetNetwork
from ray.rllib.execution.metric_ops import StandardMetricsReporting
from ray.rllib.agents.dqn.learner_thread import LearnerThread
from ray.rllib.agents.trainer import with_common_config
from ray.rllib.agents.trainer_template import build_trainer

logger = logging.getLogger(__name__)

from .policy import Policy
from .replay import ReplayActor

OPTIMIZER_SHARED_CONFIGS = [
    'buffer_size', 'prioritized_replay', 'prioritized_replay_alpha',
    'prioritized_replay_beta', 'prioritized_replay_eps',
    'rollout_fragment_length', 'train_batch_size', 'learning_starts'
]

DEFAULT_CONFIG = with_common_config({
    'framework': 'tfe',
    'eager_tracing': True,
    'use_state_preprocessor': True,
    # === Model ===
    'twin_q': True,
    # RLlib model options for the Q function(s).
    'model': {
        'custom_model': 'SACIQN',
        'custom_model_config': {
            'encoder': {},
            'actor': {},
            'q': {},
            'temperature': {}
        },
        'max_seq_len': 100   # required by DynamicTFPolicy
    },
    'N': 8,
    'N_PRIME': 8,
    'K': 32,
    'KAPPA': 1,

    # === Learning ===
    'schedule_tec': False,
    'target_entropy_coef': .5,
    'n_step': 3,
    'max_step': 5,
    'data_augmentation': False,
    'dr_coef': 1,
    'kl_coef': 1,
    'epsilon_greedy': False,
    'reward_scale': 1.,
    'reward_clip': 10.,
    'prior_lr': 1e-3,
    'entropy_v': True,
    'n_actions':1,
    'reward_entropy': False,
    'reward_prior': False,
    'temp_type': 'log',
    'max_temp': 1,
    'min_temp': -.3,
    'gamma2': .995,

    'learning_starts': int(2.5e4),
    'train_batch_size': 64,
    'rollout_fragment_length': 50, # TODO: try tune this 
    'target_network_update_freq': 96000, # training samples: 64 * 1500
    'timesteps_per_iteration': 25000,   # logging interval
    # If set, this will fix the ratio of replayed from a buffer and learned
    # on timesteps to sampled from an environment and stored in the replay
    # buffer timesteps. Otherwise, replay will proceed as fast as possible.
    'training_intensity': None,

    # === Replay buffer ===
    # Size of the replay buffer. Note that if async_updates is set, then
    # each worker will have a replay buffer of this size.
    'buffer_size': int(2.5e5),
    # If True prioritized replay buffer will be used.
    'prioritized_replay': False,
    'prioritized_replay_alpha': 0.6,
    'prioritized_replay_beta': 0.4,
    'prioritized_replay_eps': 1e-6,
    'prioritized_replay_beta_annealing_timesteps': 2000000,
    'final_prioritized_replay_beta': 0.4,
    # Whether to compute priorities on workers.
    'worker_side_prioritization': True,
    # Whether to LZ4 compress observations
    'compress_observations': False,

    # === Optimization ===
    'optimization': {
        'schedule_lr': False,
        'actor_lr': 3e-4,
        'critic_lr': 3e-4,
        'temp_lr': 3e-4,
        'epsilon': 1e-7,
    },
    'optimizer': {
        'max_weight_sync_delay': 400,   # timestep, sync weights with workers
        'num_replay_buffer_shards': 1,
        'debug': False
    },
    # If not None, clip gradients during optimization at this value.
    'grad_clip': None,

    # === Parallelism ===
    # Whether to use a GPU for local optimization.
    'num_gpus': 1,
    # Number of workers for collecting samples with. This only makes sense
    # to increase if your environment is particularly slow to sample, or if
    # you"re using the Async or Ape-X optimizers.
    'num_workers': 6,
    "num_envs_per_worker": 1,
    # Whether to allocate GPUs for workers (if > 0).
    'num_gpus_per_worker': 0,
    # Whether to allocate CPUs for workers (if > 0).
    'num_cpus_per_worker': 1,
    'num_cpus_for_driver': 1,
    # Prevent iterations from going lower than this time span.
    'min_iter_time_s': 30,
    
    'preprocessor_pref': None,

    # Use a Beta-distribution instead of a SquashedGaussian for bounded,
    # continuous action spaces (not recommended, for debugging only).
    '_use_beta_distribution': False,

    # DEPRECATED VALUES (set to -1 to indicate they have not been overwritten
    # by user's config). If we don't set them here, we will get an error
    # from the config-key checker.
    'grad_norm_clipping': DEPRECATED_VALUE,
})

def validate_config(config):
    if config['model'].get('custom_model'):
        logger.warning(
            'Setting use_state_preprocessor=True since a custom model '
            'was specified.')
        config['use_state_preprocessor'] = True

    if config.get('grad_norm_clipping', DEPRECATED_VALUE) != DEPRECATED_VALUE:
        deprecation_warning('grad_norm_clipping', 'grad_clip')
        config['grad_clip'] = config.pop('grad_norm_clipping')

    if config['grad_clip'] is not None and config['grad_clip'] <= 0.0: 
        raise ValueError('`grad_clip` value must be > 0.0!')

class UpdateWorkerWeights:
    def __init__(self, learner_thread, workers, max_weight_sync_delay):
        self.learner_thread = learner_thread
        self.workers = workers
        self.steps_since_update = collections.defaultdict(int)
        self.max_weight_sync_delay = max_weight_sync_delay
        self.weights = None

    def __call__(self, item: ("ActorHandle", SampleBatchType)):
        actor, batch = item
        self.steps_since_update[actor] += batch.count
        if self.steps_since_update[actor] >= self.max_weight_sync_delay:
            # Note that it's important to pull new weights once
            # updated to avoid excessive correlation between actors.
            if self.weights is None or self.learner_thread.weights_updated:
                self.learner_thread.weights_updated = False
                # print('weight', len(self.workers.local_worker().get_weights()['default_policy']))
                self.weights = ray.put(
                    self.workers.local_worker().get_weights())
            actor.set_weights.remote(self.weights, _get_global_vars())
            self.steps_since_update[actor] = 0

def apex_execution_plan(workers: WorkerSet, config: dict):
    # Create a number of replay buffer actors.
    num_replay_buffer_shards = config["optimizer"]["num_replay_buffer_shards"]
    replay_actors = create_colocated(ReplayActor, [
        num_replay_buffer_shards,
        config["learning_starts"],
        config["buffer_size"],
        config["train_batch_size"],
        config["prioritized_replay_alpha"],
        config["prioritized_replay_beta"],
        config["prioritized_replay_eps"],
        config['data_augmentation'],
    ], num_replay_buffer_shards)

    # Start the learner thread.
    learner_thread = LearnerThread(workers.local_worker())
    learner_thread.start()

    # Update experience priorities post learning.
    def update_prio_and_stats(item: ("ActorHandle", dict, int)):
        actor, prio_dict, count = item
        if config['prioritized_replay']:
            actor.update_priorities.remote(prio_dict)
        metrics = _get_shared_metrics()
        # Manually update the steps trained counter since the learner thread
        # is executing outside the pipeline.
        metrics.counters[STEPS_TRAINED_COUNTER] += count
        # metrics.timers["learner_dequeue"] = learner_thread.queue_timer
        # metrics.timers["learner_grad"] = learner_thread.grad_timer
        # metrics.timers["learner_overall"] = learner_thread.overall_timer

    # We execute the following steps concurrently:
    # (1) Generate rollouts and store them in our replay buffer actors. Update
    # the weights of the worker that generated the batch.
    rollouts = ParallelRollouts(workers, mode="async", num_async=2)
    store_op = rollouts \
        .for_each(StoreToReplayBuffer(actors=replay_actors))
    # Only need to update workers if there are remote workers.
    if workers.remote_workers():
        store_op = store_op.zip_with_source_actor() \
            .for_each(UpdateWorkerWeights(
                learner_thread, workers,
                max_weight_sync_delay=(
                    config["optimizer"]["max_weight_sync_delay"])
            ))

    # (2) Read experiences from the replay buffer actors and send to the
    # learner thread via its in-queue.
    replay_op = Replay(actors=replay_actors, num_async=4) \
        .zip_with_source_actor() \
        .for_each(Enqueue(learner_thread.inqueue))

    # (3) Get priorities back from learner thread and apply them to the
    # replay buffer actors.
    update_op = Dequeue(
            learner_thread.outqueue, check=learner_thread.is_alive) \
        .for_each(update_prio_and_stats) \
        .for_each(UpdateTargetNetwork(
            workers, config["target_network_update_freq"],
            by_steps_trained=True))

    # Execute (1), (2), (3) asynchronously as fast as possible. Only output
    # items from (3) since metrics aren't available before then.
    merged_op = Concurrently(
        [store_op, replay_op, update_op], mode="async", output_indexes=[2])

    # Add in extra replay and learner metrics to the training result.
    def add_apex_metrics(result):
        replay_stats = ray.get(replay_actors[0].stats.remote(
            config["optimizer"].get("debug")))
        result["info"].update({
            "learner_queue": learner_thread.learner_queue_size.stats(),
            "learner": copy.deepcopy(learner_thread.stats),
            "replay_shard_0": replay_stats,
        })
        return result

    selected_workers = workers.remote_workers()[:1]

    return StandardMetricsReporting(
        merged_op, workers, config,
        selected_workers=selected_workers).for_each(add_apex_metrics)


Agent = build_trainer(
    name='Agent',
    default_config=DEFAULT_CONFIG,
    default_policy=Policy,
    validate_config=validate_config,
    execution_plan=apex_execution_plan
)

def getstate(self):
    state = {}
    if hasattr(self, "workers"):
        state["worker"] = self.workers.local_worker().save()
    if hasattr(self, "optimizer") and hasattr(self.optimizer, "save"):
        state["optimizer"] = self.optimizer.save()
    return state


def setstate(self, state: dict):
    if "worker" in state:
        self.workers.local_worker().restore(state["worker"])
        remote_state = ray.put(state["worker"])
        for r in self.workers.remote_workers():
            r.restore.remote(remote_state)
    if "optimizer" in state:
        self.optimizer.restore(state["optimizer"])

Agent.__getstate__ = getstate
Agent.__setstate__ = setstate
