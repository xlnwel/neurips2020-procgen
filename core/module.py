import functools
import inspect
import tensorflow as tf
from tensorflow.keras import layers


class Module(tf.Module):
    """ This class aims to substitute keras.layers.Layer when 
    encapsulating multiple layers. It provides more fine-grained 
    output for keras.Model.summary.
    Moreover, you can now, without worries about name conflicts,
    define `self._layers`, which is used by default in `call`.
    """
    def __init__(self, name):
        self.scope_name = name
        self._is_built = False
        self._training_cls = [layers.BatchNormalization, layers.Dropout]
        name = name and name.split('/')[-1]
        super().__init__(name=name)

    def __call__(self, x=None, *args, **kwargs):
        if x is not None and not self._is_built:
            if isinstance(x, tf.Tensor):
                self._build(x.shape)
            else:
                self._build(x)
        return self._call(x, *args, **kwargs)
        
    def _build(self, input_shape):
        self.build(input_shape)
        self._is_built = True

    def build(self, *args, **kwargs):
        """ Override this if necessary """
        pass

    # @tf.Module.with_name_scope    # do not decorate with this, 
    def _call(self, *args, **kwargs):
        return self.call(*args, **kwargs)
        
    def call(self, x, training=False, training_cls=(), **kwargs):
        """ Override this if necessary """
        training_cls = set(training_cls) | set(self._training_cls)
        training_cls = tuple([c.func if isinstance(c, functools.partial) 
            else c for c in training_cls if inspect.isclass(c)])
        
        for l in self._layers:
            if isinstance(l, training_cls):
                x = l(x, training=training)
            else:
                x = l(x, **kwargs)
        return x

    def get_weights(self):
        return [v.numpy() for v in self.variables]

    def set_weights(self, weights):
        [v.assign(w) for v, w in zip(self.variables, weights)]

    def mlp(self, x, *args, name, **kwargs):
        if not hasattr(self, f'_{name}'):
            from nn.func import mlp
            setattr(self, f'_{name}', mlp(*args, name=name, **kwargs))
        return getattr(self, f'_{name}')(x)
