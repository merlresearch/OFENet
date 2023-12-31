# Copyright (C) 2020, 2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later
import tensorflow as tf

layers = tf.keras.layers


class MLPBlock(tf.keras.Model):
    def __init__(
        self, units, activation, kernel_initializer="glorot_uniform", batchnorm=False, trainable=True, name="mlpblock"
    ):
        super().__init__(name=name)

        self.act = activation
        self.fc = layers.Dense(units, kernel_initializer=kernel_initializer, trainable=trainable, name="fc")

        self.batchnorm = batchnorm
        if self.batchnorm:
            self.normalizer = layers.BatchNormalization()

    def call(self, inputs, training):
        features = self.fc(inputs)

        if self.batchnorm:
            features = self.normalizer(features, training=training)
        features = self.act(features)

        return features


class ResnetBlock(tf.keras.Model):
    def __init__(
        self,
        units1,
        units2,
        activation,
        kernel_initializer="glorot_uniform",
        batchnorm=False,
        trainable=True,
        name="resblock",
    ):
        super().__init__(name=name)

        self.act = activation

        self.fc1 = layers.Dense(units1, kernel_initializer=kernel_initializer, trainable=trainable, name="fc1")
        self.fc2 = layers.Dense(units2, kernel_initializer=kernel_initializer, trainable=trainable, name="fc2")

        self.batchnorm = batchnorm
        if self.batchnorm:
            self.normalizer1 = layers.BatchNormalization()
            self.normalizer2 = layers.BatchNormalization()

    def call(self, inputs, training):
        identity_map = inputs

        features = self.fc1(inputs)

        if self.batchnorm:
            features = self.normalizer1(features, training)

        features = self.act(features)
        features = self.fc2(features)

        if self.batchnorm:
            features = self.normalizer2(features, training)

        cur_dim = int(features.shape[1])
        identity_dim = int(identity_map.shape[1])

        if cur_dim > identity_dim:
            identity_map = tf.pad(identity_map, paddings=[[0, 0], [0, cur_dim - identity_dim]])
        elif cur_dim < identity_dim:
            features = tf.pad(features, paddings=[[0, 0], [0, identity_dim - cur_dim]])

        features = features + identity_map

        features = self.act(features)

        return features


class DensenetBlock(tf.keras.Model):
    def __init__(
        self, units, activation, kernel_initializer="glorot_uniform", batchnorm=False, trainable=True, name="denseblock"
    ):
        super().__init__(name=name)

        self.act = activation
        self.fc = layers.Dense(units, kernel_initializer=kernel_initializer, trainable=trainable, name="fc")

        self.batchnorm = batchnorm
        if batchnorm:
            self.normalizer = layers.BatchNormalization()

    def call(self, inputs, training):
        identity_map = inputs

        features = self.fc(inputs)

        if self.batchnorm:
            features = self.normalizer(features, training=training)

        features = self.act(features)

        features = tf.concat([features, identity_map], axis=1)

        return features
