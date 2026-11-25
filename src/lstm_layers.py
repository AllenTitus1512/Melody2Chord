"""Keras layers used by Melody2Chord's serializable masked LSTM model."""
from __future__ import annotations

import tensorflow as tf


@tf.keras.utils.register_keras_serializable(package="Melody2Chord")
class MaskAwareLSTM(tf.keras.layers.LSTM):
    """LSTM with persisted masking support for the current Keras nightly runtime."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Current Keras nightly defaults this flag to False, even though the
        # recurrent call supports a timestep mask. The subclass sets it again
        # when deserializing, so saved models ignore post-padding after reload.
        self.supports_masking = True
