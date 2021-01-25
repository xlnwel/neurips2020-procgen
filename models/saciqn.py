import numpy as np
import tensorflow as tf
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.models import ModelCatalog

from nn.func import mlp, cnn
from core.decorator import config
from core.module import Module
from utility.schedule import TFPiecewiseSchedule

Encoder = lambda config: cnn(**config)

class SACIQN(TFModelV2):
    """Extension of standard TFModel for SAC.

    Data flow:
        obs -> forward() -> model_out
        model_out -> actor() -> pi(s)
        model_out, actions -> q() -> Q(s, a)
        model_out, actions -> q2() -> Q_twin(s, a)

    Note that this class by itself is not a valid model unless you
    implement forward() in a subclass."""

    def __init__(self,
                 obs_space,
                 action_space,
                 num_outputs,
                 model_config,
                 name,
                 twin_q,
                 target_entropy_coef,
                 schedule_tec=False,
                 print_summary=False,
                 prior=False):
        """Initialize variables of this model.

        Extra model kwargs:
            actor_hidden_activation (str): activation for actor network
            actor_hiddens (list): hidden layers sizes for actor network
            critic_hidden_activation (str): activation for critic network
            critic_hiddens (list): hidden layers sizes for critic network
            twin_q (bool): build twin Q networks.
            initial_alpha (float): The initial value for the to-be-optimized
                alpha parameter (default: 1.0).

        Note that the core layers for forward() are not defined here, this
        only defines the layers for the output heads. Those layers for
        forward() should be defined in subclasses of SACModel.
        """
        super().__init__(obs_space, action_space, num_outputs,
                        model_config, name)
        model_config = model_config['custom_model_config']
        self.action_dim = action_space.n
        action_outs = q_outs = self.action_dim

        self.encoder = Encoder(model_config['encoder'])
        self.actor = Actor(model_config['actor'], action_dim=self.action_dim)
        self.q = Q(model_config['q'], action_dim=self.action_dim, name='q')

        if twin_q:
            self.q2 = Q(model_config['q'], action_dim=self.action_dim, name='q2')
        # initialize variables
        inputs = tf.keras.layers.Input(shape=obs_space.shape, dtype=np.uint8, name='obs')
        x = self.encoder(inputs)
        logits = self.actor(x)
        q = self.q(x)
        outputs = [logits, q] \
            + ([self.q2(x)] if twin_q else [])
        model = tf.keras.Model(inputs, outputs)
        if print_summary:
            # models = dict(
            #     encoder=self.encoder,
            #     actor=self.actor,
            #     q=self.q
            # )
            # from utility.display import display_model_var_info
            # display_model_var_info(models)
            model.summary(200)
        self.register_variables(
            self.encoder.variables + self.actor.variables + self.q.variables
        )
        temp_config = model_config['temperature']
        if temp_config['temp_type'] == 'constant':
            self.temp = temp_config['value']
        elif temp_config['temp_type'] == 'schedule':
            from tensorflow.keras.optimizers.schedules import PiecewiseConstantDecay
            boundaries = [8.3e4]
            values = [.01, .005]
            self.temp = PiecewiseConstantDecay(boundaries, values)
        else:
            raise ValueError(f'Unknown temperature ({temp_config["temp_type"]})')
        if twin_q:
            self.register_variables(self.q2.variables)

        if schedule_tec:
            self.target_entropy_coef = TFPiecewiseSchedule(target_entropy_coef)
        else:
            self.target_entropy_coef = target_entropy_coef
        self.target_entropy = np.log(self.action_dim, dtype=np.float32)
        if prior:
            prior = np.ones(action_space.n, dtype=np.float32)
            prior /= np.sum(prior)
            self.prior = tf.Variable(prior, trainable=False, name='prior')
            self.register_variables([self.prior])

    def forward(self, input_dict, state, seq_lens):
        obs = input_dict['obs']
        training = input_dict.get('is_training', False)
        x = self.encoder(obs, training=training)
        return x, state

    def policy_variables(self):
        return self.actor.variables

    def critic_variables(self):
        q_vars = self.encoder.variables + self.q.variables 
        if hasattr(self, 'q2'):
            q_vars += self.q2.variables
        return q_vars

    def temp_variables(self):
        return self.temp.variables


class Q(Module):
    @config
    def __init__(self, action_dim, name='q'):
        super().__init__(name=name)

        self._action_dim = action_dim

        """ Network definition """
        kwargs = dict(
            kernel_initializer=getattr(self, '_kernel_initializer', 'glorot_uniform'),
            activation=getattr(self, '_activation', 'relu'),
            out_dtype='float32',
        )
        self._kwargs = kwargs

        # we do not define the phi net here to make it consistent with the CNN output size
        if self._duel:
            self._v_head = mlp(
                self._units_list,
                out_size=1, 
                layer_type=self._layer_type,
                norm=self._norm,
                name=self.name+'v',
                **kwargs)
        self._a_head = mlp(
            self._units_list,
            out_size=action_dim, 
            layer_type=self._layer_type,
            norm=self._norm,
            name=self.name+('a' if self._duel else 'q'),
            **kwargs)

    @property
    def action_dim(self):
        return self._action_dim
    
    def call(self, x, n_qt=None, action=None, return_q=False):
        if n_qt is None:
            n_qt = self.K
        batch_size = tf.shape(x)[0]
        x = tf.expand_dims(x, 1)    # [B, 1, cnn.out_size]
        tau, tau_hat, qt_embed = self.quantile(n_qt, batch_size, x.shape[-1])
        x = x * qt_embed            # [B, N, cnn.out_size]
        qtv = self.qtv(x, action=action)
        if return_q:
            q = self.q(qtv, tau)
            return tau_hat, qtv, q
        else:
            return tau_hat, qtv
    
    def quantile(self, n_qt, batch_size, cnn_out_size):
        # phi network
        tau = tf.random.uniform([batch_size, n_qt+1, 1], 
            minval=0, maxval=1, dtype=tf.float32)   # [B, N, 1]
        tau = tf.sort(tau, axis=1)
        tau_hat = (tau[:, 1:] + tau[:, :-1]) / 2
        pi = tf.convert_to_tensor(np.pi, dtype=tau_hat.dtype)
        # start from 1 since degree of 0 is meaningless
        degree = tf.cast(tf.range(1, self._tau_embed_size+1), tau_hat.dtype) * pi * tau_hat
        qt_embed = tf.math.cos(degree)                  # [B, N, E]
        qt_embed = self.mlp(
            qt_embed, 
            [cnn_out_size], 
            name=self.name+'phi',
            **self._kwargs)                  # [B, N, cnn.out_size]
        return tau, tau_hat, qt_embed

    def qtv(self, x, action=None):
        if self._duel:
            v_qtv = self._v_head(x) # [B, N, 1]
            a_qtv = self._a_head(x) # [B, N, A]
            qtv = v_qtv + a_qtv - tf.reduce_mean(a_qtv, axis=-1, keepdims=True)
        else:
            qtv = self._a_head(x)   # [B, N, A]

        if action is not None:
            action = tf.expand_dims(action, axis=1)
            if len(action.shape) < len(qtv.shape):
                action = tf.one_hot(action, self._action_dim, dtype=qtv.dtype)
            qtv = tf.reduce_sum(qtv * action, axis=-1)      # [B, N]
            
        return qtv

    def q(self, qtv, tau_range):
        diff = tau_range[:, 1:] - tau_range[:, :-1]
        tf.debugging.assert_greater_equal(diff, 0.)
        if len(qtv.shape) < len(diff.shape):
            diff = tf.squeeze(diff, -1)
        q = tf.reduce_sum(diff * qtv, axis=1)               # [B, A] / [B]

        return q

class Actor(tf.Module):
    def __init__(self, config, action_dim, name='actor'):
        super().__init__(name=name)
        
        self._layers = mlp(
            **config,
            out_size=action_dim,
            name=name)

    def __call__(self, x, *args, **kwargs):
        x = self._layers(x, *args, **kwargs)

        return x

    def train_step(self, x, *args, **kwargs):
        x = self._layers(x, *args, **kwargs)
        probs = tf.nn.softmax(x)
        logps = tf.math.log(tf.maximum(probs, 1e-8))    # bound logps to avoid numerical instability
        return x, probs, logps


# Register model in ModelCatalog
ModelCatalog.register_custom_model("SACIQN", SACIQN)
