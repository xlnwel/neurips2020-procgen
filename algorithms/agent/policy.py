import numpy as np
import tensorflow as tf
import ray
from ray.rllib.models.tf.tf_action_dist import Categorical
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.tf_policy_template import build_tf_policy
from ray.rllib.models.catalog import ModelCatalog

from utility.rl_utils import n_step_target, qr_loss
from utility.data_augmentation import pad_crop

STEPS = "steps"
PRIO_WEIGHTS = "weights"

def build_model(policy, obs_space, action_space, config):
    model_config = config['model']
    encoder_config = model_config['custom_model_config']['encoder']
    num_outputs = compute_out_size(encoder_config)
    
    policy.model = ModelCatalog.get_model_v2(
        obs_space=obs_space,
        action_space=action_space,
        num_outputs=num_outputs,
        model_config=model_config,
        framework='tfe',
        name=model_config['custom_model'].lower(),
        twin_q=config['twin_q'], 
        target_entropy_coef=config['target_entropy_coef'],
        schedule_tec=config['schedule_tec'],
        print_summary=policy.config['worker_index'] == 0,
        prior=policy.config['epsilon_greedy'],
    )
    policy.target_model = ModelCatalog.get_model_v2(
        obs_space=obs_space,
        action_space=action_space,
        num_outputs=num_outputs,
        model_config=model_config,
        framework='tfe',
        name=model_config['custom_model'].lower(),
        twin_q=config['twin_q'], 
        target_entropy_coef=config['target_entropy_coef'],
        schedule_tec=config['schedule_tec'],
    )

    policy.reward_scale = policy.config.get('reward_scale', 1)
    policy.reward_clip = policy.config.get('reward_clip', None)

    return policy.model

def compute_out_size(config):
    if config['out_size']:
        return config['out_size']
    filters = config['filters'] if 'filters' in config else config['block_out_filters']
    return (64 / 2**len(filters))**2 * filters[-1]

def action_sampler(policy, model, obs_batch, *, explore=True, **kwargs):
    x, policy._state_out = model({
        "obs": obs_batch,
        "is_training": False,
    }, [], None)
    n_actions = policy.config.get('n_actions')
    num_workers = policy.config['num_workers']
    wid = policy.config['worker_index']
    layer_type = policy.config['model']['custom_model_config']['actor']['layer_type']
    exp_flag = wid > 0
    B = x.shape.as_list()[0]
    n = wid-1  # wid=0 is the local worker
    
    act_temp = 1 if exp_flag else .5

    logits_list = []
    # the main effect of the following loop is to increase tpf...
    if exp_flag and n_actions > 1:
        for _ in range(n_actions):
            if layer_type == 'noisy':
                logits = model.actor(x, reset=exp_flag, noisy=exp_flag) / act_temp
            else:
                logits = model.actor(x) / act_temp
            logits_list.append(logits)
        i = tf.random.uniform((B,), 0, n_actions, dtype=tf.int32)
        i2 = tf.range(B, dtype=i.dtype)
        idx = tf.stack([i, i2], axis=-1)
        logits = tf.gather_nd(logits_list, idx)
        dist = Categorical(logits, model=model)

        action = dist.sample()
    else:
        if layer_type == 'noisy':
            logits = model.actor(x, reset=exp_flag, noisy=exp_flag) / act_temp
        else:
            logits = model.actor(x) / act_temp
        dist = Categorical(logits, model=model)

        action = dist.sample()
    tf.debugging.assert_shapes([[action, (None,)]])
    
    def get_epsilon_action():
        if policy.config['epsilon_greedy'] and exp_flag:
            max_temp = policy.config['max_temp']
            min_temp = policy.config['min_temp']
            temp_fn = np.logspace if policy.config['temp_type'] == 'log' else np.linspace
            temps = temp_fn(min_temp, max_temp, num_workers * B)
            act_temp = temps[n*B:(n+1)*B]
            act_temp = np.expand_dims(act_temp, axis=-1)
            dist = Categorical(logits / act_temp, model=model)
            rand_act = dist.sample()
            epsilon = .3
            eps_action = tf.where(
                tf.random.uniform(action.shape, 0, 1) < epsilon,
                rand_act, action)
            return eps_action
        else:
            return action

    if not isinstance(explore, tf.Tensor): 
        explore = tf.constant(explore)
    action = tf.cond(explore, get_epsilon_action, lambda: action)
    if policy.config['reward_entropy']:
        logp = Categorical(logits, model=model).logp(action)
        
        model_config = policy.config['model']['custom_model_config']
        temp_config = model_config['temperature']
        temp_type = temp_config['temp_type']
        if temp_type == 'constant':
            temp = model.temp
        elif temp_type == 'schedule':
            temp = model.temp(policy.global_step)
        else:
            _, temp = model.temp()

        logp = temp * logp
    else:
        logp = np.zeros(action.shape.as_list())

    _, _, policy.v_pred = model.q(x, action=action, return_q=True)
    # we do not use&compute the log prob
    return action, logp

def _adjust_nstep(n_step, gamma, obs, actions, rewards, new_obs, dones, logp, q, max_step):
    assert not any(dones[:-1]), "Unexpected done in middle of trajectory"

    traj_length = len(rewards)
    steps = np.ones_like(rewards)
    for i in range(traj_length):
        for j in range(1, max_step):
            if i + j < traj_length:
                jth_rew = rewards[i + j] - logp[i + j]
                cum_rew = rewards[i] + gamma**j * jth_rew
                if j >= n_step and (i + j + 1 >= traj_length
                    or cum_rew + gamma**(j+1) * q[i + j + 1] <= rewards[i] + gamma**j * q[i + j]):
                    break
                new_obs[i] = new_obs[i + j]
                dones[i] = dones[i + j]
                rewards[i] += gamma**j * (rewards[i + j] - logp[i + j])
                steps[i] = j + 1
    return steps

def postprocess_trajectory(policy, batch, other_agent=None, episode=None):
    reward = batch[SampleBatch.REWARDS]
    # policy.reward_clip = max(policy.reward_clip, np.max(reward))
    reward = reward * policy.reward_scale
    reward = np.clip(reward, -policy.reward_clip, policy.reward_clip)

    batch[SampleBatch.REWARDS] = reward
    if policy.config["n_step"] > 1:
        batch.data[STEPS] = _adjust_nstep(
            policy.config["n_step"], policy.config["gamma"],
            batch[SampleBatch.CUR_OBS], batch[SampleBatch.ACTIONS],
            batch[SampleBatch.REWARDS], batch[SampleBatch.NEXT_OBS],
            batch[SampleBatch.DONES], batch[SampleBatch.ACTION_LOGP],
            batch[SampleBatch.VF_PREDS], policy.config['max_step'])
    else:
        batch.data[STEPS] = np.ones_like(batch[SampleBatch.REWARDS])


    da_config = policy.config['data_augmentation']
    if da_config:
        batch.data['obs_da'] = data_augmentation(batch[SampleBatch.CUR_OBS], da_config)
    
    return batch

def data_augmentation(img, config):
    if config['pad_crop']:
        img = pad_crop(img, **config['pad_crop'])

    return img

def compute_q_loss(x, q_fn, one_hot, q_target, N, kappa):
    tau_hat, qtvs, qs = q_fn(x, N, return_q=True)
    assert len(qtvs.shape) == len(one_hot.shape)
    qtv = tf.reduce_sum(qtvs * one_hot, axis=-1, keepdims=True)  # [B, N, 1]
    td_error, q_loss = qr_loss(qtv, q_target, tau_hat, kappa=kappa, return_error=True)
    td_error = tf.abs(td_error)
    td_error = tf.reduce_mean(td_error, axis=(1, 2))

    q_loss = tf.reduce_mean(q_loss)

    return qtvs, qs, td_error, q_loss

def sac_loss(policy, model, _, train_batch):
    N = policy.config['N']
    N_PRIME = policy.config['N_PRIME']
    kappa = policy.config['KAPPA']
    obs = train_batch[SampleBatch.CUR_OBS]
    action = train_batch[SampleBatch.ACTIONS]
    obs_next = train_batch[SampleBatch.NEXT_OBS]
    reward = train_batch[SampleBatch.REWARDS] 
    done = train_batch[SampleBatch.DONES]
    steps = train_batch[STEPS]
    model_config = policy.config['model']['custom_model_config']
    temp_config = model_config['temperature']
    layer_type = model_config['actor']['layer_type']
    temp_type = temp_config['temp_type']
    twin_q = policy.config['twin_q']

    one_hot = tf.one_hot(action, depth=model.action_dim)
    one_hot_ext = tf.expand_dims(one_hot, axis=1)

    temp = model.temp

    discount = 1.0 - tf.cast(done, tf.float32)
    reward = reward[:, None]
    discount = discount[:, None]
    gamma = policy.config["gamma"]

    x_next, _ = policy.target_model({
        "obs": obs_next,
        "is_training": False,
    }, [], None)
    if layer_type == 'noisy':
        _, act_probs_next, act_logps_next = policy.target_model.actor.train_step(
            x_next, reset=False, noisy=False)
    else:
        _, act_probs_next, act_logps_next = policy.target_model.actor.train_step(x_next)
    act_probs_next = tf.expand_dims(act_probs_next, axis=1)
    act_logps_next = tf.expand_dims(act_logps_next, axis=1)
    # Target Q-values.
    _, qtv_next = policy.target_model.q(
        x_next, N_PRIME)
    value_next = tf.reduce_sum(act_probs_next 
        * (qtv_next - temp * act_logps_next), axis=-1)
    q_target = n_step_target(reward, value_next, discount, gamma, steps)
    q_target = tf.expand_dims(q_target, axis=1)
    q_target = tf.stop_gradient(q_target)
    if twin_q:
        gamma2 = policy.config["gamma2"]
        _, qtv_next2 = policy.target_model.q2(
            x_next, N_PRIME)
        value_next2 = tf.reduce_sum(act_probs_next 
            * (qtv_next2 - temp * act_logps_next), axis=-1)
        # qtv_next = (qtv_next + qtv_next2) / 2.
        q_target2 = n_step_target(reward, value_next2, discount, gamma2, steps)
        q_target2 = tf.expand_dims(q_target2, axis=1)
        q_target2 = tf.stop_gradient(q_target2)

    tf.debugging.assert_shapes([
        [act_probs_next, (None, 1, model.action_dim)],
        [act_logps_next, (None, 1, model.action_dim)],
        [qtv_next, (None, N_PRIME, model.action_dim)],
        [value_next, (None, N_PRIME)],
        [reward, (None, 1)],
        [discount, (None, 1)],
        [q_target, (None, 1, N_PRIME)]
    ])
        
    x, _ = model({
        "obs": obs,
        "is_training": True,
    }, [], None)
    qtvs, qs, td_error, q_loss = compute_q_loss(
        x, model.q, one_hot_ext, q_target, N, kappa)
    tf.debugging.assert_shapes([
        [one_hot, (None, model.action_dim)],
        [qs, (None, model.action_dim)],
        [q_loss, (None)],
    ])
    if twin_q:
        qs2, td_error2, q_loss2 = compute_q_loss(
            x, model.q2, one_hot_ext, q_target2, N, kappa)
        qs = (qs + qs2) / 2.
        td_error = (td_error + td_error2) / 2.
        critic_loss = (q_loss + q_loss2) / 2.
    else:
        critic_loss = q_loss

    # actor losses.
    x_no_grad = tf.stop_gradient(x)
    qs = tf.stop_gradient(qs)
    temp_no_grad = tf.stop_gradient(temp)
    def compute_actor_loss(x, qs):
        act_logits, act_probs, act_logps = model.actor.train_step(x)
        entropy = -tf.reduce_sum(act_probs * act_logps, axis=-1)
        q = tf.reduce_sum(act_probs * qs, axis=-1)
        actor_loss = - tf.reduce_mean(
            temp_no_grad * entropy + q)
        return act_logits, act_probs, actor_loss, q, entropy
    act_logits, act_probs, actor_loss, q, entropy = compute_actor_loss(x_no_grad, qs)
    # if da_config:
    #     x_da = tf.stop_gradient(x_da)
    #     actor_loss_da, _, _ = compute_actor_loss(x_da, qs)
    #     actor_loss = (actor_loss + policy.config['dr_coef'] * actor_loss_da) / (1 + policy.config['dr_coef'])
    tf.debugging.assert_shapes([
        [entropy, (None,)],
        [q, (None,)]
    ])
    
    if policy.config['epsilon_greedy']:
        act_probs = tf.reduce_mean(act_probs, 0)
        model.prior.assign_add(policy.config['prior_lr'] * (act_probs - model.prior))

    policy.temp_type = temp_type
    # for gradients function
    policy.actor_loss = actor_loss
    policy.critic_loss = critic_loss
    policy.td_error = td_error
    # for stats function
    policy.stats_terms = dict(
        reward_max=tf.reduce_max(reward),
        reward_min=tf.reduce_min(reward),
        q_max=tf.reduce_max(q),
        q_min=tf.reduce_min(q),
        q_std=tf.math.reduce_std(q),
        q=tf.reduce_mean(q),
        td_error=tf.reduce_mean(td_error),
        act_logits_max=tf.reduce_max(act_logits),
        act_logits_min=tf.reduce_min(act_logits),
        act_logits_std=tf.math.reduce_std(act_logits),
        act_logits=tf.reduce_mean(act_logits),
        actor_loss=actor_loss,
        critic_loss=critic_loss,
        entropy=tf.reduce_mean(entropy),
    )
    if model.encoder.deter_stoch:
        state = model.encoder.state
        policy.stats_terms.update({
            'ds_deter': tf.reduce_mean(state.deter),
            'ds_mean': tf.reduce_mean(state.mean),
            'ds_std': tf.reduce_mean(state.std),
            'ds_stoch': tf.reduce_mean(state.stoch),
        })
    if model.encoder.belief:
        state = model.encoder.belief_state
        policy.stats_terms.update({
            'belief_deter': tf.reduce_mean(state.deter),
            'belief_mean': tf.reduce_mean(state.mean),
            'belief_std': tf.reduce_mean(state.std),
            'belief_stoch': tf.reduce_mean(state.stoch),
        })
    # for i in range(model.q.action_dim):
    #     policy.stats_terms[f'prior_{i}'] = model.prior[i]
    # if policy.config['optimization']['schedule_lr']:
    #     policy.stats_terms.update(dict(
    #         actor_lr=policy._actor_lr(policy._actor_optimizer.iterations),
    #         critic_lr=policy._critic_lr(policy._critic_optimizer.iterations),
    #         temp_lr=policy._temp_lr(policy._temp_optimizer.iterations),
    #     ))

    # in a custom apply op we handle the losses separately, but return them
    # combined in one loss for now
    policy.global_step.assign_add(1)

    loss = actor_loss + critic_loss
    return loss


def gradients_fn(policy, optimizer, loss):
    tape = optimizer.tape

    def compute_grads(loss, t_list, clip):
        grads = tape.gradient(loss, t_list)
        if clip is not None:
            grads = [tf.clip_by_value(g, -clip, clip) for g in grads]
        return list(zip(grads, t_list))

    clip = policy.config['grad_clip']

    actor_vars = policy.model.policy_variables()
    actor_grads_and_vars = compute_grads(
        policy.actor_loss, actor_vars, clip)
    
    critic_vars = policy.model.critic_variables()
    critic_grads_and_vars = compute_grads(
        policy.critic_loss, critic_vars, clip)

    # Save grads and vars for later use in `build_apply_op`.
    policy._actor_grads_and_vars = actor_grads_and_vars
    policy._critic_grads_and_vars = critic_grads_and_vars

    grads_and_vars = (
        actor_grads_and_vars 
        + critic_grads_and_vars)
        
    if policy.temp_type == 'variable':
        temp_vars = policy.model.temp_variables()
        temp_grads_and_vars = compute_grads(
            policy.temp_loss, temp_vars, clip)
        policy._temp_grads_and_vars = temp_grads_and_vars
        grads_and_vars += temp_grads_and_vars

    return grads_and_vars


def apply_gradients(policy, optimizer, grads_and_vars):
    policy._actor_optimizer.apply_gradients(policy._actor_grads_and_vars)
    policy._critic_optimizer.apply_gradients(policy._critic_grads_and_vars)
    if policy.temp_type == 'variable':
        policy._temp_optimizer.apply_gradients(policy._temp_grads_and_vars)


def stats(policy, train_batch):
    return policy.stats_terms
    

class ActorCriticOptimizerMixin:
    def __init__(self, config):
        # - Create global step for counting the number of update operations.
        self.global_step = tf.Variable(0, trainable=False, name="global_step")
        opt_config = config["optimization"]
        epsilon = opt_config["epsilon"]
        if opt_config['schedule_lr']:
            from utility.schedule import TFPiecewiseSchedule
            from tensorflow.keras.optimizers.schedules import PiecewiseConstantDecay
            # self._actor_lr = actor_lr = TFPiecewiseSchedule([(4e4, opt_config["actor_lr"]), (1e5, 1e-4)])
            # self._critic_lr = critic_lr = TFPiecewiseSchedule([(4e4, opt_config["critic_lr"]), (1e5, 1e-4)])
            # self._temp_lr = temp_lr = TFPiecewiseSchedule([(4e4, opt_config["temp_lr"]), (1e5, 1e-4)])
            boundaries = [8.3e4]
            values = [2.5e-4, 2e-4]
            self._actor_lr = actor_lr = PiecewiseConstantDecay(boundaries, values)
            self._critic_lr = critic_lr = PiecewiseConstantDecay(boundaries, values)
            self._temp_lr = temp_lr = PiecewiseConstantDecay(boundaries, values)
        else:
            actor_lr = opt_config["actor_lr"]
            critic_lr = opt_config["critic_lr"]
            temp_lr = opt_config["temp_lr"]
        self._actor_optimizer = tf.keras.optimizers.Adam(
            learning_rate=actor_lr, epsilon=epsilon)
        self._critic_optimizer = tf.keras.optimizers.Adam(
            learning_rate=critic_lr, epsilon=epsilon)
        self._temp_optimizer = tf.keras.optimizers.Adam(
            learning_rate=temp_lr, epsilon=epsilon)


class TargetNetworkMixin:
    def __init__(self, config):
        @tf.function
        def do_update():
            online_vars = self.model.critic_variables() + self.model.policy_variables()
            target_vars = self.target_model.critic_variables() + self.target_model.policy_variables()
            [tvar.assign(mvar) for tvar, mvar in zip(target_vars, online_vars)]

        self.update_target = do_update


def setup_early_mixins(policy, obs_space, action_space, config):
    from core.tf_config import configure_gpu
    configure_gpu()
    ActorCriticOptimizerMixin.__init__(policy, config)
    policy.action_sampler_fn = action_sampler


def setup_late_mixins(policy, obs_space, action_space, config):
    TargetNetworkMixin.__init__(policy, config)

def extra_fetches(policy):
    """Adds value function outputs to experience train_batches."""
    return {
        SampleBatch.VF_PREDS: policy.v_pred,
    }

Policy = build_tf_policy(
    name="Policy",
    get_default_config=lambda: ray.rllib.agents.sac.sac.DEFAULT_CONFIG,
    make_model=build_model,
    postprocess_fn=postprocess_trajectory,
    action_sampler_fn=action_sampler,
    extra_action_fetches_fn=extra_fetches,
    loss_fn=sac_loss,
    stats_fn=stats,
    gradients_fn=gradients_fn,
    apply_gradients_fn=apply_gradients,
    extra_learn_fetches_fn=lambda policy: {"td_error": policy.td_error},
    mixins=[
        TargetNetworkMixin, ActorCriticOptimizerMixin
    ],
    before_init=setup_early_mixins,
    after_init=setup_late_mixins,
    obs_include_prev_action_reward=False)
