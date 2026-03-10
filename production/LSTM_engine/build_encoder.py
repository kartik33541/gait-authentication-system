import tensorflow as tf
from tensorflow.keras.layers import Input, LSTM, Bidirectional, Dense, Dropout, BatchNormalization, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
import tensorflow.keras.backend as K


def get_encoder(input_shape=(256, 6), embedding_dim=256):

    inputs = Input(shape=input_shape, name="gait_6axis_input")

    # First temporal layer
    x = Bidirectional(LSTM(64, return_sequences=True))(inputs)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    # Second temporal layer
    x = Bidirectional(LSTM(128, return_sequences=False))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    # Dense reasoning layer
    x = Dense(
        256,
        activation="relu",
        kernel_regularizer=l2(1e-4)
    )(x)

    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    # Raw embedding
    embeddings = Dense(
        embedding_dim,
        activation=None,
        kernel_regularizer=l2(1e-4),
        name="raw_embedding"
    )(x)

    # L2 normalized embedding (for cosine similarity)
    normalized_embeddings = Lambda(
        lambda z: K.l2_normalize(z, axis=1),
        name="l2_norm_embedding"
    )(embeddings)

    encoder_model = Model(inputs, normalized_embeddings, name="Siamese_LSTM_Encoder")

    return encoder_model