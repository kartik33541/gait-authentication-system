import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import Callback
import tensorflow.keras.backend as K

# Import custom modules
from dataset_loader import load_all_users, generate_siamese_pairs
from build_encoder import get_encoder

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "siamese_lstm.weights.h5")
ENCODER_SAVE_PATH = os.path.join(BASE_DIR, "production_encoder.keras")

EPOCHS = 30
BATCH_SIZE = 1024  # Large batch size for CPU efficiency 
MARGIN = 2.0  

def euclidean_distance(vects):
    x, y = vects
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))

def contrastive_loss(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    loss_match = y_true * K.square(y_pred)
    loss_imposter = (1 - y_true) * K.square(K.maximum(MARGIN - y_pred, 0))
    return K.mean(loss_match + loss_imposter)

class SmartCosineStopping(Callback):
    def __init__(self, val_pairs, val_labels, encoder, patience=5, warmup=5):
        super().__init__()
        self.val_pairs = val_pairs
        self.val_labels = val_labels
        self.encoder = encoder
        self.patience = patience
        self.warmup = warmup
        self.best_gap = -1.0
        self.wait = 0

    def on_epoch_end(self, epoch, logs=None):
        eval_size = min(500, len(self.val_labels))
        idx = np.random.choice(len(self.val_labels), eval_size, replace=False)
        
        vec_a = self.encoder.predict(self.val_pairs[idx, 0], verbose=0)
        vec_b = self.encoder.predict(self.val_pairs[idx, 1], verbose=0)

        cos_sims = np.sum(vec_a * vec_b, axis=1)
        avg_match = np.mean(cos_sims[self.val_labels[idx] == 1])
        avg_imposter = np.mean(cos_sims[self.val_labels[idx] == 0])
        gap = avg_match - avg_imposter

        print(f"\n📊 Epoch {epoch+1} -> Gap: {gap:.4f} (Match: {avg_match:.3f} | Imposter: {avg_imposter:.3f})")

        if gap > self.best_gap:
            self.best_gap = gap
            self.wait = 0
            self.model.save_weights(MODEL_SAVE_PATH)
            self.encoder.save(ENCODER_SAVE_PATH)
            print("   🏆 New optimal model saved.")
        elif epoch >= self.warmup:
            self.wait += 1
            if self.wait >= self.patience:
                print("🛑 Early stopping triggered. Generalization peaked.")
                self.model.stop_training = True

def build_siamese_model(input_shape):
    encoder = get_encoder(input_shape=input_shape, embedding_dim=256)
    
    input_a = Input(shape=input_shape, name="left_walk")
    input_b = Input(shape=input_shape, name="right_walk")
    
    vec_a = encoder(input_a)
    vec_b = encoder(input_b)
    
    distance = Lambda(euclidean_distance, name="distance_calculator")([vec_a, vec_b])
    siamese_net = Model(inputs=[input_a, input_b], outputs=distance)
    
    siamese_net.compile(loss=contrastive_loss, optimizer=Adam(learning_rate=0.003))
    return siamese_net, encoder

def main():
    print("🚀 Initiating Fast Siamese Training Protocol\n")
    
    X_data, y_labels, _ = load_all_users()
    if len(X_data) == 0:
        print("❌ CRITICAL: No data loaded. Exiting.")
        return
        
    input_shape = (256, 6)
    pairs, labels = generate_siamese_pairs(X_data, y_labels)
    
    split_idx = int(len(pairs) * 0.9)
    train_x, val_x = pairs[:split_idx], pairs[split_idx:]
    train_y, val_y = labels[:split_idx], labels[split_idx:]

    print("🥞 Formatting Data for CPU (Wait a few seconds)...")
    # tf.data pipelines dramatically speed up CPU training
    train_ds = tf.data.Dataset.from_tensor_slices(((train_x[:, 0], train_x[:, 1]), train_y))
    train_ds = train_ds.shuffle(buffer_size=2000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    val_ds = tf.data.Dataset.from_tensor_slices(((val_x[:, 0], val_x[:, 1]), val_y))
    val_ds = val_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    
    siamese_model, base_encoder = build_siamese_model(input_shape)
    callbacks = [SmartCosineStopping(val_x, val_y, base_encoder)]
    
    print(f"\n🔥 Starting Training Engine on {len(train_x)} pairs...")
    print("Note: The first epoch progress bar will appear shortly.")
    
    siamese_model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        verbose=1, # Guaranteed to show progress bar now that batch is fixed
        callbacks=callbacks
    )

if __name__ == "__main__":
    main()