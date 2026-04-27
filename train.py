# stage 3
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def build_autoencoder(input_dim, bottleneck_dim=64, hidden_dim=128):
    inputs = keras.Input(shape=(input_dim,))
    h = layers.Dense(hidden_dim, activation="relu")(inputs)
    z = layers.Dense(bottleneck_dim)(h)
    h2 = layers.Dense(hidden_dim, activation="relu")(z)
    out = layers.Dense(input_dim, activation="sigmoid")(h2)

    autoencoder = keras.Model(inputs, out)
    encoder = keras.Model(inputs, z)

    autoencoder.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="mse",
    )
    return autoencoder, encoder


def train(X, epochs=300, batch_size=None, bottleneck_dim=64):
    if batch_size is None:
        batch_size = len(X)

    autoencoder, encoder = build_autoencoder(
        input_dim=X.shape[1],
        bottleneck_dim=bottleneck_dim,
    )

    autoencoder.fit(
        X,
        X,
        epochs=epochs,
        batch_size=batch_size,
        verbose=0,
        callbacks=[
            keras.callbacks.LambdaCallback(
                on_epoch_end=lambda epoch, logs: print(
                    f"epoch {epoch:4d}  loss {logs['loss']:.6f}"
                )
                if (epoch % 25 == 0 or epoch == epochs - 1)
                else None
            )
        ],
    )

    embeddings = encoder.predict(X, verbose=0)
    return autoencoder, encoder, embeddings


if __name__ == "__main__":
    data = np.load("cache/features.npz", allow_pickle=True)
    X = data["feature_matrix"].astype("float32")
    game_ids = data["game_ids"]
    game_names = data["game_names"]

    print(f"Training on {X.shape[0]} games × {X.shape[1]} features")
    print(f"GPUs available: {len(tf.config.list_physical_devices('GPU'))}")

    autoencoder, encoder, embeddings = train(X, epochs=300)

    np.savez(
        "cache/embeddings.npz",
        embeddings=embeddings,
        game_ids=game_ids,
        game_names=game_names,
    )
    autoencoder.save("cache/autoencoder.keras")
    encoder.save("cache/encoder.keras")

    print(f"Saved embeddings: {embeddings.shape}")
