import os
import json
import tensorflow as tf
from functools import wraps
import os
import pickle as pkl

from ..base import (BaseModel, DtypeMixin,
                                     is_param_name)


class TensorFlowModel(BaseModel, DtypeMixin):
    def __init__(self, model_path='tf_model/', paths=None,
                 tf_session_config=None, tf_saver_params=None, json_params=None,
                 *args, **kwargs):
        super(TensorFlowModel, self).__init__(*args, **kwargs)
        self._model_dirpath = None
        self._model_filepath = None
        self._params_filepath = None
        self._random_state_filepath = None
        self._train_summary_dirpath = None
        self._val_summary_dirpath = None
        self._tf_meta_graph_filepath = None
        self.update_working_paths(model_path=model_path, paths=paths)

        # self._tf_session_config = tf_session_config or tf.compat.v1.ConfigProto()
        self.tf_saver_params = tf_saver_params or {}
        self.json_params = json_params or {}
        self.json_params.setdefault('sort_keys', True)
        self.json_params.setdefault('indent', 4)
        self.initialized_ = False

        self._tf_session = None
        self._tf_saver = None
        self._tf_merged_summaries = None
        self._tf_train_writer = None
        self._tf_val_writer = None

        # Saves tf variables
        self._checkpoint = None

    @staticmethod
    def compute_working_paths(model_path):
        """
        Parameters
        ----------
        model_path : str
            Model dirpath (should contain slash at the end) or filepath
        """
        head, tail = os.path.split(model_path)
        if not head: head = '.'
        if not head.endswith('/'): head += '/'
        if not tail: tail = 'model'

        paths = {}
        paths['model_dirpath'] = head
        paths['model_filepath'] = os.path.join(paths['model_dirpath'], tail)
        paths['params_filepath'] = os.path.join(paths['model_dirpath'], 'params.json')
        paths['random_state_filepath'] = os.path.join(paths['model_dirpath'], 'random_state.json')
        paths['train_summary_dirpath'] = os.path.join(paths['model_dirpath'], 'logs/train')
        paths['val_summary_dirpath'] = os.path.join(paths['model_dirpath'], 'logs/val')
        paths['tf_meta_graph_filepath'] = paths['model_filepath'] + '.meta'
        return paths

    def update_working_paths(self, model_path=None, paths=None):
        paths = paths or {}
        if not paths:
            paths = TensorFlowModel.compute_working_paths(model_path=model_path)
        for k, v in paths.items():
            setattr(self, '_{0}'.format(k), v)

    def _make_tf_model(self):
        raise NotImplementedError('`_make_tf_model` is not implemented')

    def _init_tf_ops(self):
        """Initialize all TF variables and Saver"""
        init_op = tf.compat.v1.global_variables_initializer()
        self._tf_session.run(init_op)
        self._tf_saver = tf.compat.v1.train.Saver(**self.tf_saver_params)

    def _init_tf_writers(self):
        self._tf_merged_summaries = tf.compat.v1.summary.merge_all()
        self._tf_train_writer = tf.compat.v1.summary.FileWriter(self._train_summary_dirpath,
                                                      self._tf_graph)
        self._tf_val_writer = tf.compat.v1.summary.FileWriter(self._val_summary_dirpath,
                                                    self._tf_graph)

    def _save_model(self, global_step=None):
        # # (recursively) create all folders needed
        for dirpath in (self._train_summary_dirpath, self._val_summary_dirpath):
            if not os.path.exists(dirpath):
                os.makedirs(dirpath)
        #
        # # save params
        # params = self.get_params(deep=False)
        # params = self._serialize(params)
        # params['__class_name__'] = self.__class__.__name__
        # with open(self._params_filepath, 'w') as params_file:
        #     json.dump(params, params_file, **self.json_params)
        #
        # # dump random state if needed
        # if self.random_seed is not None:
        #     random_state = self._rng.get_state()
        #     with open(self._random_state_filepath, 'w') as random_state_file:
        #         json.dump(random_state, random_state_file)
        #
        #

        # write tensorflow variables
        # self.save_tf_variables()
        # tf_vars = {k: v for k, v in self.__dict__.items() if 'tensorflow' in str(type(v)) and 'Variable' in str(type(v))}
        # for k in tf_vars.keys():
        #     self.__dict__[k] = None # can't pickle tf variables or checkpoint

        # checkpoint = self.__dict__['_checkpoint']
        # self.__dict__['_checkpoint'] = None

        with open(os.path.join(self._model_filepath), "wb") as f:
            pkl.dump(self, f)

        # self.__dict__['_checkpoint'] = checkpoint

    @classmethod
    def load_model(cls, model_path):
        # paths = TensorFlowModel.compute_working_paths(model_path)
        #
        # # load params
        # with open(paths['params_filepath'], 'r') as params_file:
        #     params = json.load(params_file)
        # class_name = params.pop('__class_name__')
        # if class_name != cls.__name__:
        #     raise RuntimeError("attempt to load {0} with class {1}".format(class_name, cls.__name__))
        # d = {k: params[k] for k in params if is_param_name(k)}
        # model = cls(paths=paths, **d)
        # params = model._deserialize(params)
        # model.set_params(**params)  # set attributes and deserialized params
        #
        # # restore random state if needed
        # if os.path.isfile(model._random_state_filepath):
        #     with open(model._random_state_filepath, 'r') as random_state_file:
        #         random_state = json.load(random_state_file)
        #     model._rng.set_state(random_state)
        #
        # # restore all tensorflow variables
        # model.load_tf_variables()


        with open(os.path.join(model_path, 'model'), "rb") as f:
            model = pkl.load(f)

        # model.load_tf_variables()

        return model


    def save_tf_variables(self):
        if self._checkpoint is None:
            tf_variables = {k: v for k, v in self.__dict__.items() if 'tensorflow' in str(type(v)) and 'Variable' in str(type(v))}
            self._checkpoint = tf.train.Checkpoint(**tf_variables)
        self._checkpoint.write(self._model_filepath)

    def load_tf_variables(self):
        tf_variables = {k: v for k, v in self.__dict__.items() if 'tensorflow' in str(type(v)) and 'Variable' in str(type(v))}
        self._checkpoint = tf.train.Checkpoint(**tf_variables)
        self._checkpoint.restore(self._model_filepath)


    def _fit(self, X, X_val=None, *args, **kwargs):
        """Class-specific `fit` routine."""
        raise NotImplementedError('`fit` is not implemented')

    def init(self):
        if not self.initialized_:
            self.initialized_ = True
            self._save_model()
        return self

    def fit(self, X, X_val=None, *args, **kwargs):
        """Fit the model according to the given training data."""
        self.initialized_ = True
        self._fit(X, X_val=X_val, *args, **kwargs)
        self._save_model()
        return self

    def get_tf_params(self, scope=None):
        """Get tf params of the model.

        Returns
        -------
        params : dict[str] = np.ndarray
            Evaluated parameters of the model.
        """
        weights = {}
        for var in tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope=scope):
            key = var.name
            if scope and scope in key:
                key = key.replace(scope, '')
            if key.startswith('/'):
                key = key[1:]
            if key.endswith(':0'):
                key = key[:-2]
            weights[key] = var.eval()
        return weights


if __name__ == '__main__':
    # run corresponding tests
    from ..utils.testing import run_tests
    from .tests import test_tf_model as t
    run_tests(__file__, t)
