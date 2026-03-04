import tensorflow as tf
from tensorflow.keras.layers import Input, LSTM, Bidirectional, Dense, Dropout, BatchNormalization, Lambda
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K

def get_encoder(input_shape=(256, 6), embedding_dim=256): 
    """
    Builds a deep Siamese LSTM network optimized for 6-axis gait biomechanics.
    Speed optimized for CPU training via LSTM unrolling and BatchNormalization.
    """
    inputs = Input(shape=input_shape, name="gait_6axis_input")

    # Layer 1: unroll=True forces the CPU to process this without looping overhead
    x = Bidirectional(LSTM(64, return_sequences=True, unroll=True))(inputs)
    x = BatchNormalization()(x)  
    x = Dropout(0.3)(x)          

    # Layer 2
    x = Bidirectional(LSTM(128, return_sequences=False, unroll=True))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    # Layer 3: Dense Reasoning
    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)  
    x = Dropout(0.3)(x)

    # Layer 4: The 256-D Embedding Space
    embeddings = Dense(embedding_dim, activation=None, name="raw_embedding")(x)

    # Normalization 4: L2 Normalization (Forces outputs onto a hypersphere for Cosine distance)
    normalized_embeddings = Lambda(lambda z: K.l2_normalize(z, axis=1), name="l2_norm_embedding")(embeddings)

    encoder_model = Model(inputs, normalized_embeddings, name="Siamese_LSTM_Encoder")
    return encoder_model