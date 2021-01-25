import re
import numpy as np
import tensorflow as tf
from tensorflow.keras.mixed_precision import experimental as prec

def select_optimizer(name):
    opts = dict(
        adam=tf.keras.optimizers.Adam,
        rmsprop=tf.keras.optimizers.RMSprop,
    )
    return opts[name.lower()]


class Optimizer(tf.Module):
    def __init__(self, name, models, lr, clip_norm=None, weight_decay=None, wdpattern=r'.*', return_grads=False, **kwargs):
        self._models = models if isinstance(models, (list, tuple)) else [models]
        self._clip_norm = clip_norm
        self._weight_decay = weight_decay
        self._wdpattern = wdpattern
        self._opt = select_optimizer(name)(lr, **kwargs)
        self._return_grads = return_grads
        # useful for mixed precision training on GPUs to
        # avoid numerical underflow caused by using float16 gradients
        self._mpt = prec.global_policy().should_cast_variables
        if self._mpt:
            print('Mixed precision training will be performed')
            self._opt = prec.LossScaleOptimizer(self._opt, 'dynamic')
        # we do not initialize variables here, as models may not be initialized at this point
        self._variables = None
    
    def set_weights(self, weights):
        self._opt.set_weights(weights)

    def get_weights(self):
        return self._opt.get_weights()

    def __call__(self, tape, loss, output_gradients=None):
        if self._variables is None:
            variables = [m.trainable_variables for m in self._models]
            self._variables = tf.nest.flatten(variables)
        assert len(loss.shape) == 0, loss.shape
        if self._mpt:
            with tape:
                loss = self._opt.get_scaled_loss(loss)
        grads = tape.gradient(loss, self._variables, output_gradients=output_gradients)
        if self._mpt:
            grads = self._opt.get_unscaled_gradients(grads)
        norm = tf.linalg.global_norm(grads)
        if self._clip_norm:
            grads, _ = tf.clip_by_global_norm(grads, self._clip_norm, norm)
        if self._weight_decay:
            context = tf.distribute.get_replica_context()
            context.merge_call(self._apply_weight_decay)
        self._opt.apply_gradients(zip(grads, self._variables))

        if self._return_grads:
            return norm, {v.name: g for v, g in zip(self._variables, grads)}
        else:
            return norm
    
    def _apply_weight_decay(self, strategy):
        for var in self._variables:
            if re.search(self._wdpattern, var.name):
                strategy.extended.update(var, lambda var: self._weight_decay * var)
