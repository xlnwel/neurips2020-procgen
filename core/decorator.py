from functools import wraps


def config(init_fn):
    def wrapper(self, config, *args, **kwargs):
        _config_attr(self, config)

        init_fn(self, *args, **kwargs)

    return wrapper

def step_track(learn_log):
    @wraps(learn_log)
    def wrapper(self, step):
        if step > self.env_step:
            self.env_step = step
        self.train_step += learn_log(self, step)

    return wrapper

def override(cls):
    @wraps(cls)
    def check_override(method):
        if method.__name__ not in dir(cls):
            raise NameError("{} does not override any method of {}".format(
                method, cls))
        return method

    return check_override

""" Functions used to print model variables """                    
def _config_attr(obj, config):
    for k, v in config.items():
        if not k.isupper():
            k = f'_{k}'
        if isinstance(v, str):
            try:
                v = float(v)
            except:
                if v.lower() == 'none':
                    v = None
        if isinstance(v, float) and v == int(v):
            v = int(v)
        setattr(obj, k, v)
