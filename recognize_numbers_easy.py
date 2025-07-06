import os
import tensorflow as tf
print(tf.__version__)

(training_images, training_labels), _ = tf.keras.datasets.mnist.load_data(path='mnist.npz')


# Normalize pixel values
training_images = training_images / 255.0

def create_and_compile_model():
    model = tf.keras.models.Sequential([ 
		tf.keras.Input(shape=(28,28)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation=tf.nn.relu),
        tf.keras.layers.Dense(10, activation=tf.nn.softmax)
    ]) 


    model.compile(
		optimizer='adam',
		loss='sparse_categorical_crossentropy',
		metrics=['accuracy']
	)

    return model

untrained_model = create_and_compile_model()

predictions = untrained_model.predict(training_images[:5], verbose=False)

print(f"predictions have shape: {predictions.shape}")


class EarlyStoppingCallback(tf.keras.callbacks.Callback):

    def on_epoch_end(self, epoch, logs=None):
        if logs['accuracy'] >= 0.98:
            self.model.stop_training = True

            print("\nReached 98% accuracy so cancelling training!")


def train_mnist(training_images, training_labels):
    model = create_and_compile_model()
    return model.fit(training_images, training_labels, epochs=10, callbacks=[EarlyStoppingCallback()])


training_history = train_mnist(training_images, training_labels)

print("Training history keys:", training_history.history.keys())
for key, values in training_history.history.items():
    print(f"{key}: {values}")


# Load test data
_, (test_images, test_labels) = tf.keras.datasets.mnist.load_data(path='mnist.npz')
test_images = test_images / 255.0

# Create and train a new model instance
model = create_and_compile_model()
model.fit(training_images, training_labels, epochs=10, callbacks=[EarlyStoppingCallback()], verbose=False)

# Evaluate on test data
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print(f"\nTest accuracy: {test_acc:.4f}")