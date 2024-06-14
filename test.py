import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt

# Load the beans dataset
dataset, info = tfds.load('beans', with_info=True, as_supervised=True)

# Split the dataset into training, validation, and test sets
train_ds, val_ds, test_ds = dataset['train'], dataset['validation'], dataset['test']

# Print dataset information
print(info)


# Define preprocessing function
def preprocess(image, label):
    image = tf.cast(image, tf.float32) / 255.0  # Normalize the image to [0,1] range
    return image, label

# Apply the preprocessing function to the datasets
train_ds = train_ds.map(preprocess).shuffle(1000).batch(32).prefetch(tf.data.experimental.AUTOTUNE)
val_ds = val_ds.map(preprocess).batch(32).prefetch(tf.data.experimental.AUTOTUNE)
test_ds = test_ds.map(preprocess).batch(32).prefetch(tf.data.experimental.AUTOTUNE)


# Define a simple CNN model
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(500, 500, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(train_ds, validation_data=val_ds, epochs=10)
