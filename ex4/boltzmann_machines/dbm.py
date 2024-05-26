import numpy as np
import tensorflow as tf
from tensorflow.core.framework import summary_pb2
import tensorflow_probability as tfp

from .ebm import EnergyBasedModel
from .layers import BernoulliLayer
from .utils import (make_list_from, write_during_training,
                   batch_iter, epoch_iter,
                   log_sum_exp, log_diff_exp, log_mean_exp, log_std_exp)


class DBM(EnergyBasedModel):
    """Deep Boltzmann Machine with EM-like learning algorithm
    based on PCD and mean-field variational inference [1].

    Parameters
    ----------
    rbms : [BaseRBM]
        Array of already pretrained RBMs going from visible units
        to the most hidden ones.
    n_particles : positive int
        Number of "persistent" Markov chains (i.e., "negative" or "fantasy" "particles").
    v_particle_init : None or (n_particles, n_visible) np.ndarray
        If provided, initialize visible particle from this matrix,
        otherwise initialize using resp. stochastic layer initializer.
    h_particles_init : None or iterable of None or (n_particles, n_hiddens[i]) np.ndarray
        Same semantics as for `v_particle_init`, but for hidden particles for all layers
    n_gibbs_steps : positive int or iterable
        Number of Gibbs steps for PCD. Values are updated after each epoch.
    max_mf_updates : positive int
        Maximum number of mean-field updates per weight update.
    mf_tol : positive float
        Mean-field tolerance.
    learning_rate, momentum : positive float or iterable
        Gradient descent parameters. Values are updated after each epoch.
    max_epoch : positive int
        Train till this epoch.
    batch_size : positive int
        Input batch size for training. Total number of training examples should
        be divisible by this number.
    l2 : non-negative float
        L2 weight decay coefficient.
    max_norm : positive float
        Maximum norm constraint. Might be useful to use for this model instead
        of L2 weight decay as recommended in [3].
    sample_v_states : bool
        Whether to sample visible states, or to use probabilities
        w/o sampling.
    sample_h_states : (n_layers,) bool
        Whether to sample hidden states, or to use probabilities
        w/o sampling.
    sparsity_target : float in (0, 1) or iterable
        Desired probability of hidden activation (for different hidden layers).
    sparsity_cost : non-negative float or iterable
        Controls the amount of sparsity penalty (for different hidden layers).
    sparsity_damping : float in (0, 1)
        Decay rate for hidden activations probs.
    train_metrics_every_iter, val_metrics_every_epoch : positive int
        Control frequency of logging progress
    verbose : bool
        Whether to display progress during training.
    save_after_each_epoch : bool
        If False, save model only after the whole training is complete.
    display_filters : non-negative int
        Number of weights filters to display during training (in TensorBoard).
    display_particles : non-negative int
        Number of hidden activations to display during training (in TensorBoard).
    v_shape : (H, W) or (H, W, C) positive integer tuple
        Shape for displaying filters during training. C should be in {1, 3, 4}.

    References
    ----------
    [1] R. Salakhutdinov and G. Hinton. Deep boltzmann machines.
        In AISTATS, pp. 448-455. 2009
    [2] Salakhutdinov, R. Learning Deep Boltzmann Machines, MATLAB code.
        url: https://www.cs.toronto.edu/~rsalakhu/DBM.html
    [3] I.J. Goodfellow, A. Courville, and Y. Bengio. Joint training deep
        boltzmann machines for classification. arXiv preprint arXiv:1301.3568.
        2013.
    [4] G. Monvaton and K.-R. Mueller. Deep boltzmann machines and
        centering trick. In Neural Networks: Tricks of the trade,
        pp. 621-637, Springer, 2012.
    [5] G. Hinton and R. Salakhutdinov. A better way to pretrain Deep
        Boltzmann Machines. In Advances in Neural Information Processing
        Systems, pp. 2447-2455, 2012.
    """
    def __init__(self, rbms=None,
                 n_particles=100, v_particle_init=None, h_particles_init=None,
                 n_gibbs_steps=5, max_mf_updates=10, mf_tol=1e-7,
                 learning_rate=0.0005, momentum=0.9, max_epoch=10, batch_size=100,
                 l2=0., max_norm=np.inf,
                 sample_v_states=True, sample_h_states=None,
                 sparsity_target=0.1, sparsity_cost=0., sparsity_damping=0.9,
                 train_metrics_every_iter=10, val_metrics_every_epoch=1,
                 verbose=False, save_after_each_epoch=True,
                 display_filters=0, display_particles=0, v_shape=(28, 28),
                 model_path='dbm_model/', *args, **kwargs):
        super(DBM, self).__init__(model_path=model_path, *args, **kwargs)
        self.n_layers_ = len(rbms) if rbms is not None else None
        self.n_visible_ = None
        self.n_hiddens_ = []


        self.n_particles = n_particles
        self._v_particle_init = v_particle_init
        self._h_particles_init = h_particles_init

        self.n_gibbs_steps = make_list_from(n_gibbs_steps)
        self.max_mf_updates = max_mf_updates
        self.mf_tol = mf_tol

        self.learning_rate = make_list_from(learning_rate)
        self.momentum = make_list_from(momentum)
        self.max_epoch = max_epoch
        self.batch_size = batch_size
        self.l2 = l2
        self.max_norm = max_norm

        self.sample_v_states = sample_v_states
        self.sample_h_states = sample_h_states or [True] * self.n_layers_

        self.sparsity_target = make_list_from(sparsity_target)
        self.sparsity_cost = make_list_from(sparsity_cost)
        if self.n_layers_ is not None and self.n_layers_ > 1:
            for x in (self.sparsity_target, self.sparsity_cost):
                if len(x) == 1:
                    x *= self.n_layers_
        self.sparsity_damping = sparsity_damping

        self.train_metrics_every_iter = train_metrics_every_iter
        self.val_metrics_every_epoch = val_metrics_every_epoch
        self.verbose = verbose
        self.save_after_each_epoch = save_after_each_epoch

        for nh in self.n_hiddens_:
            assert nh >= display_filters
        self.display_filters = display_filters

        assert display_particles <= self.n_particles
        self.display_particles = display_particles

        self.v_shape = v_shape
        if len(self.v_shape) == 2:
            self.v_shape = (self.v_shape[0], self.v_shape[1], 1)

        # additional attributes
        self.epoch_ = 0
        self.iter_ = 0
        self.n_samples_generated_ = 0

        # tf constants
        self._n_visible = None
        self._n_hiddens = []
        self._n_particles = None
        self._max_mf_updates = None
        self._mf_tol = None

        self._sparsity_targets = []
        self._sparsity_costs = []
        self._sparsity_damping = None

        self._batch_size = None
        self._l2 = None
        self._max_norm = None
        self._N = None
        self._M = None

        # tf input data
        # self._learning_rate = None
        # self._momentum = None
        # self._n_gibbs_steps = None
        # self._X_batch = None
        # self._delta_beta = None
        # self._n_ais_runs = None

        # tf vars
        self._W = []
        self._vb = None
        self._hb = []

        self._dW = []
        self._dvb = None
        self._dhb = []

        self._mu = []
        self._mu_new = []
        self._q_means = []
        self._mu_means = []

        self._v = None
        self._v_new = None
        self._H = []
        self._H_new = []

        # tf operations
        # self._train_op = None
        # self._transform_op = None
        # self._msre = None
        # self._reconstruction = None
        # self._n_mf_updates = None
        # self._sample_v = None
        # self._log_Z = None
        # self._log_proba = None

        self.load_rbms(rbms)




    def load_rbms(self, rbms):
        if rbms is not None:

            self._rbms = rbms

            # create some shortcuts
            self.n_layers_ = len(self._rbms)
            self.n_visible_ = self._rbms[0].n_visible
            self.n_hiddens_ = [rbm.n_hidden for rbm in self._rbms]

            # extract weights and biases
            self._W_init, self._vb_init, self._hb_init = [], [], []
            for i in range(self.n_layers_):
                self._W_init.append(self._rbms[i]._W)
                self._vb_init.append(self._rbms[i]._vb)
                self._hb_init.append(self._rbms[i]._hb)

            # collect resp. layers of units
            self._v_layer = self._rbms[0]._v_layer
            self._h_layers = [rbm._h_layer for rbm in self._rbms]

            # ... and update their dtypes
            self._v_layer.dtype = self.dtype
            for h in self._h_layers:
                h.dtype = self.dtype

            self._make_constants()
            self._make_vars()




    def _make_constants(self):
        with tf.compat.v1.name_scope('constants'):
            self._n_visible = tf.constant(self.n_visible_, dtype=tf.int32, name='n_visible')
            if self._n_hiddens is None:
                self._n_hiddens = []
            for i in range(self.n_layers_):
                T = tf.constant(self.n_hiddens_[i], dtype=tf.int32, name='n_hidden')
                self._n_hiddens.append(T)
            self._n_particles = tf.constant(self.n_particles, dtype=tf.int32, name='n_particles')
            self._max_mf_updates = tf.constant(self.max_mf_updates,
                                               dtype=tf.int32, name='max_mf_updates')
            self._mf_tol = tf.constant(self.mf_tol, dtype=self._tf_dtype, name='mf_tol')

            for i in range(self.n_layers_):
                T = tf.constant(self.sparsity_target[i], dtype=self._tf_dtype, name='sparsity_target')
                self._sparsity_targets.append(T)
                C = tf.constant(self.sparsity_cost[i], dtype=self._tf_dtype, name='sparsity_cost')
                self._sparsity_costs.append(C)
            self._sparsity_damping = tf.constant(self.sparsity_damping, dtype=self._tf_dtype, name='sparsity_damping')

            self._batch_size = tf.constant(self.batch_size, dtype=tf.int32, name='batch_size')
            self._l2 = tf.constant(self.l2, dtype=self._tf_dtype, name='L2_coef')
            self._max_norm = tf.constant(self.max_norm, dtype=self._tf_dtype, name='max_norm_coef')
            self._N = tf.cast(self._batch_size, dtype=self._tf_dtype, name='N')
            self._M = tf.cast(self._n_particles, dtype=self._tf_dtype, name='M')


    def _make_vars(self):
        # compose weights and biases of DBM from trained RBMs' ones
        # and account double-counting evidence problem [1]
        W_init, hb_init = [], []
        vb_init = self._vb_init[0]
        for i in range(self.n_layers_):
            W = self._W_init[i]
            vb = self._vb_init[i]
            hb = self._hb_init[i]

            # halve weights and biases of intermediate RBMs
            if 0 < i < self.n_layers_ - 1:
                W *= 0.5
                vb *= 0.5
                hb *= 0.5

            # initialize weights
            W_init.append(W)

            # initialize hidden biases as average of respective biases
            # of respective RBMs, as in [2]
            if i == 0:
                hb_init.append(0.5 * hb)
            else:  # i > 0
                hb_init[i - 1] += 0.5 * vb
                hb_init.append(0.5 * hb if i < self.n_layers_ - 1 else hb)

        # initialize weights and biases
        with tf.compat.v1.name_scope('weights'):
            t = tf.constant(vb_init, dtype=self._tf_dtype, name='vb_init')
            self._vb = tf.Variable(t, dtype=self._tf_dtype, name='vb')
            tf.compat.v1.summary.histogram('vb_hist', self._vb)

            for i in range(self.n_layers_):
                T = tf.constant(W_init[i], dtype=self._tf_dtype, name='W_init')
                W = tf.Variable(T, dtype=self._tf_dtype, name='W')
                self._W.append(W)
                tf.compat.v1.summary.histogram('W_hist', W)

            for i in range(self.n_layers_):
                t = tf.constant(hb_init[i],  dtype=self._tf_dtype, name='hb_init')
                hb = tf.Variable(t,  dtype=self._tf_dtype, name='hb')
                self._hb.append(hb)
                tf.compat.v1.summary.histogram('hb_hist', hb)

        # visualize filters
        if self.display_filters:
            with tf.compat.v1.name_scope('filters_visualization'):
                W = self._W[0]
                for i in range(self.n_layers_):
                    if i > 0:
                        W = tf.matmul(W, self._W[i])
                    W_display = tf.transpose(a=W, perm=[1, 0])
                    W_display = tf.reshape(W_display, [self.n_hiddens_[i], self.v_shape[2],
                                                       self.v_shape[0], self.v_shape[1]])
                    W_display = tf.transpose(a=W_display, perm=[0, 2, 3, 1])
                    tf.summary.image('W_filters', W_display, max_outputs=self.display_filters)

        # initialize gradients accumulators
        with tf.compat.v1.name_scope('grads_accumulators'):
            t = tf.zeros(vb_init.shape, dtype=self._tf_dtype, name='dvb_init')
            self._dvb = tf.Variable(t, name='dvb')
            tf.compat.v1.summary.histogram('dvb_hist', self._dvb)

            for i in range(self.n_layers_):
                T = tf.zeros(W_init[i].shape, dtype=self._tf_dtype, name='dW_init')
                dW = tf.Variable(T, name='dW')
                tf.compat.v1.summary.histogram('dW_hist', dW)
                self._dW.append(dW)

            for i in range(self.n_layers_):
                t = tf.zeros(hb_init[i].shape, dtype=self._tf_dtype, name='dhb_init')
                dhb = tf.Variable(t, name='dhb')
                tf.compat.v1.summary.histogram('dhb_hist', dhb)
                self._dhb.append(dhb)

        # initialize variational parameters
        with tf.compat.v1.name_scope('variational_params'):
            for i in range(self.n_layers_):
                t = tf.zeros([self._batch_size, self.n_hiddens_[i]], dtype=self._tf_dtype)
                mu = tf.Variable(t, name='mu')
                t_new = tf.zeros([self._batch_size, self.n_hiddens_[i]], dtype=self._tf_dtype)
                mu_new = tf.Variable(t_new, name='mu_new')
                tf.compat.v1.summary.histogram('mu_hist', mu)
                self._mu.append(mu)
                self._mu_new.append(mu_new)

        # initialize running means of hidden activations means
        with tf.compat.v1.name_scope('hidden_means_accumulators'):
            for i in range(self.n_layers_):
                T = tf.Variable(tf.zeros([self.n_hiddens_[i]], dtype=self._tf_dtype), name='q_means')
                self._q_means.append(T)
                S = tf.Variable(tf.zeros([self.n_hiddens_[i]], dtype=self._tf_dtype), name='mu_means')
                self._mu_means.append(S)

        # initialize negative particles
        with tf.compat.v1.name_scope('negative_particles'):
            if self._v_particle_init is not None:
                t = tf.constant(self._v_particle_init, dtype=self._tf_dtype, name='v_init')
            else:
                t = self._v_layer.init(batch_size=self._n_particles)
            self._v = tf.Variable(t, dtype=self._tf_dtype, name='v')
            t_new = self._v_layer.init(batch_size=self._n_particles)
            self._v_new = tf.Variable(t_new, dtype=self._tf_dtype, name='v_new')

            for i in range(self.n_layers_):
                with tf.compat.v1.name_scope('h_particle'):
                    if self._h_particles_init is not None:
                        q = tf.constant(self._h_particles_init[i],
                                        shape=[self.n_particles, self.n_hiddens_[i]],
                                        dtype=self._tf_dtype, name='h_init')
                    else:
                        q = self._h_layers[i].init(batch_size=self._n_particles)
                    h = tf.Variable(q, dtype=self._tf_dtype, name='h')
                    q_new = self._h_layers[i].init(batch_size=self._n_particles)
                    h_new = tf.Variable(q_new, dtype=self._tf_dtype, name='h_new')
                    self._H.append(h)
                    self._H_new.append(h_new)


    def gibbs_step(self, v, H, H_new, update_v=True, sample=True):

        # update first hidden layer

        T = tf.matmul(v, self._W[0])
        if self.n_layers_ >= 2:
            T += tf.matmul(a=H[1], b=self._W[1], transpose_b=True)
        H_new[0] = self._h_layers[0].activation(T, self._hb[0])
        if sample and self.sample_h_states[0]:
            with tf.compat.v1.name_scope('sample_h0_hat_given_v_h1'):
                H_new[0] = self._h_layers[0].sample(means=H_new[0])

        # update the intermediate hidden layers if any
        for i in range(1, self.n_layers_ - 1):
            T1 = tf.matmul(H_new[i - 1], self._W[i])
            T2 = tf.matmul(a=H[i + 1], b=self._W[i + 1], transpose_b=True)
            H_new[i] = self._h_layers[i].activation(T1 + T2, self._hb[i])
            if sample and self.sample_h_states[i]:
                H_new[i] = self._h_layers[i].sample(means=H_new[i])

        # update last hidden layer
        if self.n_layers_ >= 2:
            T = tf.matmul(H_new[-2], self._W[-1])
            H_new[-1] = self._h_layers[-1].activation(T, self._hb[-1])
            if sample and self.sample_h_states[-1]:
                H_new[-1] = self._h_layers[-1].sample(means=H_new[-1])

        # update visible layer if needed
        if update_v:
            T = tf.matmul(a=H_new[0], b=self._W[0], transpose_b=True)
            v_new = self._v_layer.activation(T, self._vb)
            if sample and self.sample_v_states:
                v_new = self._v_layer.sample(means=v_new)
        else:
            v_new = v

        return v_new, H, H_new


    def _make_gibbs_step(self, v, H, v_new, H_new, update_v=True, sample=True):
        """Compute one Gibbs step."""
        with tf.compat.v1.name_scope('gibbs_step'):

            # update first hidden layer
            with tf.compat.v1.name_scope('means_h0_hat_given_v_h1'):
                T = tf.matmul(v, self._W[0])
                if self.n_layers_ >= 2:
                    T += tf.matmul(a=H[1], b=self._W[1], transpose_b=True)
                H_new[0] = self._h_layers[0].activation(T, self._hb[0])
            if sample and self.sample_h_states[0]:
                with tf.compat.v1.name_scope('sample_h0_hat_given_v_h1'):
                    H_new[0] = self._h_layers[0].sample(means=H_new[0])

            # update the intermediate hidden layers if any
            for i in range(1, self.n_layers_ - 1):
                with tf.compat.v1.name_scope('means_h{0}_hat_given_h{1}_hat_h{2}'.format(i, i - 1, i + 1)):
                    T1 = tf.matmul(H_new[i - 1], self._W[i])
                    T2 = tf.matmul(a=H[i + 1], b=self._W[i + 1], transpose_b=True)
                    H_new[i] = self._h_layers[i].activation(T1 + T2, self._hb[i])
                if sample and self.sample_h_states[i]:
                    with tf.compat.v1.name_scope('sample_h{0}_hat_given_h{1}_hat_h{2}'.format(i, i - 1, i + 1)):
                        H_new[i] = self._h_layers[i].sample(means=H_new[i])

            # update last hidden layer
            if self.n_layers_ >= 2:
                with tf.compat.v1.name_scope('means_h{0}_hat_given_h{1}_hat'.format(self.n_layers_ - 1, self.n_layers_ - 2)):
                    T = tf.matmul(H_new[-2], self._W[-1])
                    H_new[-1] = self._h_layers[-1].activation(T, self._hb[-1])
                if sample and self.sample_h_states[-1]:
                    with tf.compat.v1.name_scope('sample_h{0}_hat_given_h{1}_hat'.format(self.n_layers_ - 1, self.n_layers_ - 2)):
                        H_new[-1] = self._h_layers[-1].sample(means=H_new[-1])

            # update visible layer if needed
            if update_v:
                with tf.compat.v1.name_scope('means_v_hat_given_h0_hat'):
                    T = tf.matmul(a=H_new[0], b=self._W[0], transpose_b=True)
                    v_new = self._v_layer.activation(T, self._vb)
                if sample and self.sample_v_states:
                    with tf.compat.v1.name_scope('sample_v_hat_given_h_hat'):
                        v_new = self._v_layer.sample(means=v_new)

        return v, H, v_new, H_new

    # has side effects, changes self._mu
    def meanfield_update(self, x_batch):

        """Run mean-field updates for current mini-batch"""
        with tf.compat.v1.name_scope('mean_field'):
            # initialize mu_new using approximate inference
            # as suggested in [1]
            T = None
            for i in range(self.n_layers_):
                if i == 0:
                    T = 2. * tf.matmul(x_batch, self._W[0])
                else:
                    T = tf.matmul(T, self._W[i])
                    if i < self.n_layers_ - 1:
                        T *= 2.
                T = self._h_layers[i].activation(T, self._hb[i])
                q_new = tf.identity(T, name='approx_inference')
                self._mu_new[i] = q_new


            # run mean-field updates until convergence, indicated by _mf_tol
            max_step = self._max_mf_updates
            step = 0
            mu = self._mu
            mu_new = self._mu_new
            while step < max_step and tf.reduce_max(input_tensor=[tf.norm(tensor=u - v, ord=np.inf) for u, v in zip(mu, mu_new)]) > self._mf_tol:
                _, H, H_new = self.gibbs_step(x_batch, mu, mu_new, update_v=False, sample=False)
                mu = H_new
                mu_new = H

                step = step + 1

            n_mf_updates = step # total number of mean field updates

            for i in range(self.n_layers_):
                self._mu[i] = mu_new[i]

            return n_mf_updates, mu_new


    def particles_update(self, n_steps=None, sample=True):
        """Update negative particles by running Gibbs sampler
                for specified number of steps.
                """
        if n_steps is None:
            n_steps = self.get_current_gibbs_steps()

        v = self._v
        H = self._H
        v_new = self._v_new
        H_new = self._H_new
        for i in range(n_steps):
            v, H_new, H = self.gibbs_step(v, H, H_new, update_v=True, sample=sample)



        v_update = self._v.assign(v)
        v_new_update = self._v_new.assign(v_new)

        for i in range(self.n_layers_):
            self._H[i] = H[i]

        H_updates = [self._H[i] for i in range(self.n_layers_)]

        return v_update, H_updates



    def _apply_max_norm(self, T):
        T_norm = tf.norm(tensor=T, axis=0)
        return T * tf.minimum(T_norm, self._max_norm) / tf.maximum(T_norm, 1e-8), T_norm


    def train_op(self, x_batch):

        _learning_rate = self.get_current_learning_rate()
        _momentum = self.get_current_momentum()

        self.meanfield_update(x_batch)
        self.particles_update()

        if self.display_particles:
            with tf.compat.v1.name_scope('particles_visualization'):
                v_means, H_means = self.particles_update(sample=False)

                V = v_means[:self.display_particles, :]
                V_display = tf.reshape(V, [self.display_particles, self.v_shape[2],
                                           self.v_shape[0], self.v_shape[1]])
                V_display = tf.transpose(a=V_display, perm=[0, 2, 3, 1])
                V_display = tf.cast(V_display, tf.float32)
                tf.summary.image('visible_activations_means', V_display, max_outputs=self.display_filters)

                for i in range(self.n_layers_):
                    h_means_display = H_means[i][:, :self.display_particles]
                    h_means_display = tf.cast(h_means_display, tf.float32)
                    h_means_display = tf.expand_dims(h_means_display, 0)
                    h_means_display = tf.expand_dims(h_means_display, -1)
                    tf.summary.image('hidden_activations_means', h_means_display)

        with tf.compat.v1.name_scope('grads_estimates'):
            # visible bias
            with tf.compat.v1.name_scope('dvb'):
                dvb = tf.reduce_mean(input_tensor=x_batch, axis=0) - tf.reduce_mean(input_tensor=self._v, axis=0)

            dW = []
            # first layer of weights
            with tf.compat.v1.name_scope('dW'):
                dW_0_positive = tf.matmul(a=x_batch, b=self._mu[0], transpose_a=True) / self._N
                dW_0_negative = tf.matmul(a=self._v, b=self._H[0], transpose_a=True) / self._M
                dW_0 = (dW_0_positive - dW_0_negative) - self._l2 * self._W[0]
                dW.append(dW_0)

            # ... rest of them
            for i in range(1, self.n_layers_):
                with tf.compat.v1.name_scope('dW'):
                    dW_i_positive = tf.matmul(a=self._mu[i - 1], b=self._mu[i], transpose_a=True) / self._N
                    dW_i_negative = tf.matmul(a=self._H[i - 1], b=self._H[i], transpose_a=True) / self._M
                    dW_i = (dW_i_positive - dW_i_negative) - self._l2 * self._W[i]
                    dW.append(dW_i)

            dhb = []
            # hidden biases
            for i in range(self.n_layers_):
                with tf.compat.v1.name_scope('dhb'):
                    dhb_i = tf.reduce_mean(input_tensor=self._mu[i], axis=0) - tf.reduce_mean(input_tensor=self._H[i],
                                                                                              axis=0)
                    dhb.append(dhb_i)

            # apply sparsity targets if needed
        with tf.compat.v1.name_scope('sparsity_targets'):
            for i in range(self.n_layers_):
                q_means = tf.reduce_sum(input_tensor=self._H[i], axis=0)
                q_update = self._q_means[i].assign(self._sparsity_damping * self._q_means[i] + \
                                                   (1 - self._sparsity_damping) * q_means[i])
                mu_means = tf.reduce_sum(input_tensor=self._mu[i], axis=0)
                mu_update = self._mu_means[i].assign(self._sparsity_damping * self._mu_means[i] + \
                                                     (1 - self._sparsity_damping) * mu_means[i])
                sparsity_penalty = self._sparsity_costs[i] * (q_update - self._sparsity_targets[i])
                sparsity_penalty += self._sparsity_costs[i] * (mu_update - self._sparsity_targets[i])
                dW[i] -= sparsity_penalty
                dhb[i] -= sparsity_penalty


        # update parameters
        with tf.compat.v1.name_scope('momentum_updates'):
            with tf.compat.v1.name_scope('dvb'):
                dvb_update = self._dvb.assign(_learning_rate * (_momentum * self._dvb + dvb))
                vb_update = self._vb.assign_add(dvb_update)

            W_updates = []
            W_norms = []
            for i in range(self.n_layers_):
                with tf.compat.v1.name_scope('dW'):
                    dW_update = self._dW[i].assign(_learning_rate * (_momentum * self._dW[i] + dW[i]))
                    W_update = self._W[i] + dW_update
                    with tf.compat.v1.name_scope('max_norm'):
                        W_new, W_norm = self._apply_max_norm(W_update)
                    W_update = self._W[i].assign(W_new)
                    W_norms.append(tf.minimum(tf.reduce_max(input_tensor=W_norm), self._max_norm))
                    W_updates.append(W_update)

            hb_updates = []
            for i in range(self.n_layers_):
                with tf.compat.v1.name_scope('dhb'):
                    dhb_update = self._dhb[i].assign(
                        _learning_rate * (_momentum * self._dhb[i] + dhb[i]))
                    hb_update = self._hb[i].assign_add(dhb_update)
                    hb_updates.append(hb_update)


    def msre(self, x_batch):
        with tf.compat.v1.name_scope('mean_squared_reconstruction_error'):
            T = tf.matmul(a=self._mu[0], b=self._W[0], transpose_b=True)
            v_means = self._v_layer.activation(T, self._vb)
            v_means = tf.identity(v_means, name='x_reconstruction')
            msre = tf.reduce_mean(input_tensor=tf.square(x_batch - v_means))
            tf.compat.v1.add_to_collection('msre', msre)
            return msre



    def _unnormalized_log_prob_H0(self, x, beta):
        T1 = tf.einsum('ij,j->i', x, self._hb[0])
        T1 *= beta
        log_p = T1
        T2 = tf.matmul(x, b=self._W[0], transpose_b=True) + self._vb
        T2 *= beta
        log_p += tf.reduce_sum(input_tensor=tf.nn.softplus(T2), axis=1)
        T3 = tf.matmul(x, self._W[1]) + self._hb[1]
        T3 *= beta
        log_p += tf.reduce_sum(input_tensor=tf.nn.softplus(T3), axis=1)
        return log_p

    def _make_ais_next_sample(self, x, beta):
        def cond(step, max_step, x):
            return step < max_step

        def body(step, max_step, x):
            # v_hat <- P(v|h=x)
            T1 = tf.matmul(a=x, b=self._W[0], transpose_b=True)
            v = self._v_layer.activation(beta * T1, beta * self._vb)
            if self.sample_v_states:
                v = self._v_layer.sample(means=v)

            # h2_hat <- P(h2|h=x)
            T2 = tf.matmul(x, self._W[1])
            h2 = self._h_layers[1].activation(beta * T2, beta * self._hb[1])
            if self.sample_h_states[1]:
                h2 = self._h_layers[1].sample(means=h2)

            # x_hat <- P(h|v=v_hat, h2=h2_hat)
            T3 = tf.matmul(v, self._W[0])
            T4 = tf.matmul(a=h2, b=self._W[1], transpose_b=True)
            x_hat = self._h_layers[0].activation(beta * (T3 + T4), beta * self._hb[0])
            if self.sample_h_states[0]:
                x_hat = self._h_layers[0].sample(means=x_hat)

            return step + 1, max_step, x_hat

        _, _, x_new = tf.while_loop(cond=cond, body=body,
                                    loop_vars=[tf.constant(0),
                                               self._n_gibbs_steps,
                                               x],
                                    parallel_iterations=1,
                                    back_prop=False)
        return x_new

    def _make_ais(self):
        with tf.compat.v1.name_scope('annealed_importance_sampling'):

            # x_0 ~ Ber(0.5) of size (M, H_1)
            logits = tf.zeros([self._n_ais_runs, self._n_hiddens[0]])
            T = tfp.distributions.Bernoulli(logits=logits).sample(seed=self.make_random_seed())
            x_0 = tf.cast(T, dtype=self._tf_dtype)

            # x_1 ~ T_1(x_1 | x_0)
            x_1 = self._make_ais_next_sample(x_0, self._delta_beta)

            # -log p_0(x_1)
            log_Z = -self._unnormalized_log_prob_H0(x_1, 0.)

            def cond(log_Z, x, beta, delta_beta):
                return beta < 1. - delta_beta + 1e-5

            def body(log_Z, x, beta, delta_beta):
                # + log p_i(x_i)
                log_Z += self._unnormalized_log_prob_H0(x, beta)
                # x_{i + 1} ~ T_{i + 1}(x_{i + 1} | x_i)
                x_new = self._make_ais_next_sample(x, beta + delta_beta)
                # - log p_i(x_{i + 1})
                log_Z -= self._unnormalized_log_prob_H0(x_new, beta)
                return log_Z, x_new, beta + delta_beta, delta_beta

            log_Z, x_M, _, _ = tf.while_loop(cond=cond, body=body,
                                             loop_vars=[log_Z, x_1, self._delta_beta,
                                                                    self._delta_beta],
                                             back_prop=False,
                                             parallel_iterations=1)
            # + log p_M(x_M)
            log_Z += self._unnormalized_log_prob_H0(x_M, 1.)

            # + log(Z_0) = (V + H_1 + H_2) * log(2)
            log_Z0 = self._n_visible + self._n_hiddens[0] + self._n_hiddens[1]
            log_Z0 = tf.cast(log_Z0, dtype=self._tf_dtype)
            log_Z0 *= tf.cast(tf.math.log(2.), dtype=self._tf_dtype)
            log_Z += log_Z0

        tf.compat.v1.add_to_collection('log_Z', log_Z)


    def log_proba(self, x_batch):
        with tf.compat.v1.name_scope('log_proba'):
            n_mf_updates, mu_updates = self.meanfield_update(x_batch) # has side effects, changes self._mu

            t1 = tf.matmul(x_batch, self._W[0])
            minus_E = tf.reduce_sum(input_tensor=t1 * self._mu[0], axis=1)
            t2 = tf.matmul(self._mu[0], self._W[1])
            minus_E += tf.reduce_sum(input_tensor=t2 * self._mu[1], axis=1)
            minus_E += tf.einsum('ij,j->i', x_batch, self._vb)
            minus_E += tf.einsum('ij,j->i', self._mu[0], self._hb[0])
            minus_E += tf.einsum('ij,j->i', self._mu[1], self._hb[1])

            s1 = tf.clip_by_value(self._mu[0], 1e-7, 1. - 1e-7)
            s2 = tf.clip_by_value(self._mu[1], 1e-7, 1. - 1e-7)
            S1 = -s1 * tf.math.log(s1) - (1. - s1) * tf.math.log(1. - s1)
            S2 = -s2 * tf.math.log(s2) - (1. - s2) * tf.math.log(1. - s2)
            H = tf.reduce_sum(input_tensor=S1, axis=1) + tf.reduce_sum(input_tensor=S2, axis=1)

            log_p = minus_E + H
            return log_p

    def get_current_gibbs_steps(self):
        return self.n_gibbs_steps[min(self.epoch_, len(self.n_gibbs_steps) - 1)]

    def get_current_learning_rate(self):
        return self.learning_rate[min(self.epoch_, len(self.learning_rate) - 1)]

    def get_current_momentum(self):
        return self.momentum[min(self.epoch_, len(self.momentum) - 1)]

    def _make_tf_model(self):
        self._make_constants()
        self._make_placeholders()
        self._make_vars()

        self._make_train_op()
        self._make_sample_v()
        self._make_ais()
        self._make_log_proba()

    def _make_tf_feed_dict(self, X_batch=None, delta_beta=None, n_ais_runs=None, n_gibbs_steps=None):
        d = {}
        d['learning_rate'] = self.learning_rate[min(self.epoch_, len(self.learning_rate) - 1)]
        d['momentum'] = self.momentum[min(self.epoch_, len(self.momentum) - 1)]

        if X_batch is not None:
            d['X_batch'] = X_batch
        if delta_beta is not None:
            d['delta_beta'] = delta_beta
        if n_ais_runs is not None:
            d['n_ais_runs'] = n_ais_runs
        if n_gibbs_steps is not None:
            d['n_gibbs_steps'] = n_gibbs_steps
        else:
            d['n_gibbs_steps'] = self.n_gibbs_steps[min(self.epoch_, len(self.n_gibbs_steps) - 1)]

        # prepend name of the scope, and append ':0'
        feed_dict = {}
        for k, v in d.items():
            feed_dict['input_data/{0}:0'.format(k)] = v
        return feed_dict

    def _train_epoch(self, X):
        train_msres, train_n_mf_updates = [], []
        for X_batch in batch_iter(X, self.batch_size, verbose=self.verbose):
            self.iter_ += 1
            if self.iter_ % self.train_metrics_every_iter == 0:

                n_mf_upds, mu_updates = self.meanfield_update(x_batch=X_batch)
                msre = self.msre(X_batch)
                self.train_op(X_batch)

                train_msres.append(msre)
                train_n_mf_updates.append(n_mf_upds)
            else:
                self.train_op(X_batch)

        return (np.mean(train_msres) if train_msres else None,
                np.mean(train_n_mf_updates) if train_n_mf_updates else None)

    def _run_val_metrics(self, X_val):
        val_msres, val_n_mf_updates = [], []
        for X_vb in batch_iter(X_val, batch_size=self.batch_size):

            msre = self.msre(X_vb)
            n_mf_updates, mu_updates = self.meanfield_update(x_batch=X_vb)
            # msre, n_mf_upds = self._tf_session.run([self._msre, self._n_mf_updates],
            #                                         feed_dict=self._make_tf_feed_dict(X_vb))
            val_msres.append(msre)
            val_n_mf_updates.append(n_mf_updates)
        mean_msre = np.mean(val_msres)
        mean_n_mf_updates = np.mean(val_n_mf_updates)
        # s = summary_pb2.Summary(value=[
        #     summary_pb2.Summary.Value(tag='mean_squared_recon_error', simple_value=mean_msre),
        #     summary_pb2.Summary.Value(tag='n_mf_updates', simple_value=mean_n_mf_updates),
        # ])
        # self._tf_val_writer.add_summary(s, self.iter_)
        return mean_msre, mean_n_mf_updates

    def _fit(self, X, X_val=None, *args, **kwargs):
        # load ops requested
        # self._train_op = tf.compat.v1.get_collection('train_op')[0]
        # self._msre = tf.compat.v1.get_collection('msre')[0]
        # self._n_mf_updates = tf.compat.v1.get_collection('n_mf_updates')[0]

        # main loop
        val_msre, val_n_mf_updates = None, None
        for self.epoch_ in epoch_iter(start_epoch=self.epoch_, max_epoch=self.max_epoch,
                                      verbose=self.verbose):
            train_msre, train_n_mf_updates = self._train_epoch(X)

            # run validation metrics if needed
            if X_val is not None and self.epoch_ % self.val_metrics_every_epoch == 0:
                val_msre, val_n_mf_updates = self._run_val_metrics(X_val)

            # print progress
            if self.verbose:
                s = "epoch: {0:{1}}/{2}".format(self.epoch_, len(str(self.max_epoch)), self.max_epoch)
                if train_msre:
                    s += "; msre: {0:.5f}".format(train_msre)
                if train_n_mf_updates:
                    s += "; n_mf_upds: {0:.1f}".format(train_n_mf_updates)
                if val_msre:
                    s += "; val.msre: {0:.5f}".format(val_msre)
                if val_n_mf_updates:
                    s += "; val.n_mf_upds: {0:.1f}".format(val_n_mf_updates)
                write_during_training(s)

            # save if needed
            if self.save_after_each_epoch:
                self._save_model(global_step=self.epoch_)

    def transform(self, X, np_dtype=None):
        """Compute hidden units' (from last layer) activation probabilities."""
        np_dtype = np_dtype or self._np_dtype

        self._transform_op = tf.compat.v1.get_collection('transform_op')[0]
        G = np.zeros((len(X), self.n_hiddens_[-1]), dtype=np_dtype)
        start = 0
        for X_b in batch_iter(X, batch_size=self.batch_size,
                              verbose=self.verbose, desc='transform'):
            G_b = self._transform_op.eval(feed_dict=self._make_tf_feed_dict(X_b))
            G[start:(start + self.batch_size)] = G_b
            start += self.batch_size
        return G


    def reconstruct(self, X):
        """Compute p(v|h_0=q, h...)=p(v|h_0=q), where q=p(h_0|v=x)"""
        self._reconstruction = tf.compat.v1.get_collection('reconstruction')[0]
        X_recon = np.zeros_like(X)
        start = 0
        for X_b in batch_iter(X, batch_size=self.batch_size,
                              verbose=self.verbose, desc='reconstruction'):
            X_recon_b = self._reconstruction.eval(feed_dict=self._make_tf_feed_dict(X_b))
            X_recon[start:(start + self.batch_size)] = X_recon_b
            start += self.batch_size
        return X_recon


    def sample_v(self, n_gibbs_steps=0, save_model=False):
        """Compute visible particle activation probabilities
        after `n_gibbs_steps` chain iterations.
        """
        with tf.compat.v1.name_scope('sample_v'):
            v_update, H_updates = self.particles_update(n_steps=n_gibbs_steps)
            v_means, _ = self.particles_update(sample=False,n_steps=n_gibbs_steps)
            sample_v = self._v.assign(v_update)

        if save_model:
            self.n_samples_generated_ += n_gibbs_steps
            self._save_model()
        return sample_v


    def log_Z(self, n_betas=100, n_runs=100, n_gibbs_steps=5):
        """Estimate log partition function using Annealed Importance Sampling.
        Currently implemented only for 2-layer binary BM.
        AIS is run on a state space x = {h_1} with v and h_2
        analytically summed out, as in [1] and using formulae from [4].
        To obtain reasonable estimate, parameter `n_betas` should be at least 10000 or more.

        Parameters
        ----------
        n_betas : >1 int
            Number of intermediate distributions.
        n_runs : positive int
            Number of AIS runs.
        n_gibbs_steps : positive int
            Number of Gibbs steps per transition.

        Returns
        -------
        log_mean, (log_low, log_high) : float
            `log_mean` = log(Z_mean)
            `log_low`  = log(Z_mean - std(Z))
            `log_high` = log(Z_mean + std(Z))
        values : (`n_runs`,) np.ndarray
            All estimates.
        """
        assert self.n_layers_ == 2
        for L in [self._v_layer] + self._h_layers:
            assert isinstance(L, BernoulliLayer)

        self._log_Z = tf.compat.v1.get_collection('log_Z')[0]
        values = self._tf_session.run(self._log_Z,
                                      feed_dict=self._make_tf_feed_dict(delta_beta=1./n_betas,
                                                                        n_ais_runs=n_runs,
                                                                        n_gibbs_steps=n_gibbs_steps))

        log_mean = log_mean_exp(values)
        log_std  = log_std_exp(values, log_mean_exp_x=log_mean)
        log_high = log_sum_exp([log_std, log_mean])
        log_low  = log_diff_exp([log_std, log_mean])[0]
        return log_mean, (log_low, log_high), values


    def log_proba(self, X_test, log_Z):
        """Estimate variational lower-bound on a test set, as in [5].
        Currently implemented only for 2-layer binary BM.
        """
        assert self.n_layers_ == 2
        for L in [self._v_layer] + self._h_layers:
            assert isinstance(L, BernoulliLayer)

        self._log_proba = tf.compat.v1.get_collection('log_proba')[0]
        P = np.zeros(len(X_test))
        start = 0
        for X_b in batch_iter(X_test, batch_size=self.batch_size, verbose=self.verbose):
            P_b = self._log_proba.eval(feed_dict=self._make_tf_feed_dict(X_b))
            P[start:(start + self.batch_size)] = P_b
            start += self.batch_size
        return P - log_Z


if __name__ == '__main__':
    # run corresponding tests
    from boltzmann_machines.utils.testing import run_tests
    run_tests(__file__)
