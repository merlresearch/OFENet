# Copyright (C) 2020, 2023 Mitsubishi Electric Research Laboratories (MERL)
# Copyright (c) 2019 Kei Ohta
#
# SPDX-License-Identifier: AGPL-3.0-or-later
# SPDX-License-Identifier: MIT

# Forked from https://github.com/keiohta/tf2rl/blob/master/tf2rl/algos/ppo.py

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.layers import Dense

layers = tf.keras.layers


class GaussianActor(tf.keras.Model):
    def __init__(
        self,
        state_dim,
        action_dim,
        max_action,
        layer_units=(256, 256),
        hidden_activation="tanh",
        name="gaussian_policy",
    ):
        super().__init__(name=name)

        base_layers = []
        for cur_layer_size in layer_units:
            cur_layer = layers.Dense(cur_layer_size, activation=hidden_activation)
            base_layers.append(cur_layer)

        self.base_layers = base_layers

        self.out_mean = layers.Dense(action_dim, name="L_mean")
        # State independent log covariance
        self.out_logstd = tf.Variable(
            initial_value=-0.5 * np.ones(action_dim, dtype=np.float32), dtype=tf.float32, name="logstd"
        )

        self._max_action = max_action

        dummy_state = tf.constant(np.zeros(shape=[1, state_dim], dtype=np.float32))
        self(dummy_state)

    def _dist_from_states(self, states):
        features = states

        for cur_layer in self.base_layers:
            features = cur_layer(features)

        mu_t = self.out_mean(features)

        log_sigma_t = self.out_logstd

        dist = tfp.distributions.MultivariateNormalDiag(loc=mu_t, scale_diag=tf.exp(log_sigma_t))

        return dist

    def call(self, states):
        dist = self._dist_from_states(states)
        raw_actions = dist.sample()
        log_pis = dist.log_prob(raw_actions)

        actions = raw_actions * self._max_action
        return actions, log_pis

    def mean_action(self, states):
        dist = self._dist_from_states(states)
        raw_actions = dist.mean()
        log_pis = dist.log_prob(raw_actions)

        actions = raw_actions * self._max_action
        return actions, log_pis

    def compute_log_probs(self, states, actions):
        dist = self._dist_from_states(states)

        raw_actions = actions / self._max_action
        log_pis = dist.log_prob(raw_actions)

        return log_pis


class CriticV(tf.keras.Model):
    def __init__(self, state_dim, units, name="qf"):
        super().__init__(name=name)

        self.l1 = Dense(units[0], name="L1", activation="tanh")
        self.l2 = Dense(units[1], name="L2", activation="tanh")
        self.l3 = Dense(1, name="L2", activation="linear")

        with tf.device("/cpu:0"):
            self(tf.constant(np.zeros(shape=(1, state_dim), dtype=np.float32)))

    def call(self, inputs):
        features = self.l1(inputs)
        features = self.l2(features)
        values = self.l3(features)

        return tf.squeeze(values, axis=1)


class PPO(tf.keras.Model):
    def __init__(
        self,
        state_dim,
        action_dim,
        max_action,
        feature_extractor,
        actor_units=(64, 64),
        critic_units=(64, 64),
        lr=1e-3,
        clip_ratio=0.2,
        batch_size=64,
        discount=0.99,
        n_epoch=10,
        horizon=2048,
        lam=0.95,
        gpu=0,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.discount = discount
        self.n_epoch = n_epoch
        self.device = "/gpu:{}".format(gpu) if gpu >= 0 else "/cpu:0"
        self.horizon = horizon
        self.lam = lam
        self.clip_ratio = clip_ratio
        assert self.horizon % self.batch_size == 0, "Horizon should be divisible by batch size"

        self.actor = GaussianActor(feature_extractor.dim_state_features, action_dim, max_action, actor_units)
        self.critic = CriticV(feature_extractor.dim_state_features, critic_units)

        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

        self.ofe_net = feature_extractor

    def get_action(self, raw_state, test=False):
        assert isinstance(raw_state, np.ndarray), "Input instance should be np.ndarray, not {}".format(type(raw_state))

        is_single_input = raw_state.ndim == 1
        if is_single_input:
            raw_state = np.expand_dims(raw_state, axis=0).astype(np.float32)

        action, logp = self._get_action_body(raw_state, test)[:2]

        if is_single_input:
            return action.numpy()[0], logp.numpy()
        else:
            return action.numpy(), logp.numpy()

    def get_action_and_val(self, raw_state, test=False):
        is_single_input = raw_state.ndim == 1
        if is_single_input:
            raw_state = np.expand_dims(raw_state, axis=0).astype(np.float32)

        action, logp, v = self._get_action_logp_v_body(raw_state, test)

        if is_single_input:
            v = v[0]
            action = action[0]

        return action.numpy(), logp.numpy(), v.numpy()

    @tf.function
    def _get_action_logp_v_body(self, raw_state, test):
        action, logp = self._get_action_body(raw_state, test)[:2]
        state_feature = self.ofe_net.features_from_states(raw_state)
        v = self.critic(state_feature)
        return action, logp, v

    @tf.function
    def _get_action_body(self, state, test):
        state_feature = self.ofe_net.features_from_states(state)
        if test:
            return self.actor.mean_action(state_feature)
        else:
            return self.actor(state_feature)

    def select_action(self, raw_state):
        action, logp = self.get_action(raw_state, test=True)
        return action

    def train(self, raw_states, actions, advantages, logp_olds, returns):
        # Train actor and critic
        actor_loss, logp_news, ratio = self._train_actor_body(raw_states, actions, advantages, logp_olds)
        critic_loss = self._train_critic_body(raw_states, returns)

        # Visualize results in TensorBoard
        tf.summary.scalar(name="PPO/actor_loss", data=actor_loss)
        # tf.summary.scalar(name="PPO/logp_max",
        #                   data=np.max(logp_news))
        # tf.summary.scalar(name="PPO/logp_min",
        #                   data=np.min(logp_news))
        tf.summary.scalar(name="PPO/logp_mean", data=np.mean(logp_news))
        # tf.summary.scalar(name="PPO/adv_max",
        #                   data=np.max(advantages))
        # tf.summary.scalar(name="PPO/adv_min",
        #                   data=np.min(advantages))
        tf.summary.scalar(name="PPO/kl", data=tf.reduce_mean(logp_olds - logp_news))
        tf.summary.scalar(name="PPO/ratio", data=tf.reduce_mean(ratio))
        tf.summary.scalar(name="PPO/critic_loss", data=critic_loss)
        return actor_loss, critic_loss

    @tf.function
    def _train_actor_body(self, raw_states, actions, advantages, logp_olds):
        with tf.device(self.device):
            with tf.GradientTape() as tape:
                state_features = self.ofe_net.features_from_states(raw_states)
                logp_news = self.actor.compute_log_probs(state_features, actions)
                ratio = tf.math.exp(logp_news - tf.squeeze(logp_olds))
                min_adv = tf.clip_by_value(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * tf.squeeze(advantages)
                actor_loss = -tf.reduce_mean(tf.minimum(ratio * tf.squeeze(advantages), min_adv))
            actor_grad = tape.gradient(actor_loss, self.actor.trainable_variables)
            self.actor_optimizer.apply_gradients(zip(actor_grad, self.actor.trainable_variables))

        return actor_loss, logp_news, ratio

    @tf.function
    def _train_critic_body(self, raw_states, returns):
        with tf.device(self.device):
            # Train baseline
            with tf.GradientTape() as tape:
                state_features = self.ofe_net.features_from_states(raw_states)
                current_V = self.critic(state_features)
                td_errors = tf.squeeze(returns) - current_V
                critic_loss = tf.reduce_mean(0.5 * tf.square(td_errors))
            critic_grad = tape.gradient(critic_loss, self.critic.trainable_variables)
            self.critic_optimizer.apply_gradients(zip(critic_grad, self.critic.trainable_variables))

        return critic_loss
