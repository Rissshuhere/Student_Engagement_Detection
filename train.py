import tensorflow as tf
from tensorflow.keras import layers, models

print("Starting Student Engagement Detection")

# Dataset paths
train_path = "dataset/EED/EED/train"
test_path = "dataset/EED/EED/test"

IMG_SIZE = (128, 128)
BATCH_SIZE = 32

# Load training data
train_data = tf.keras.utils.image_dataset_from_directory(
    train_path,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True,
    validation_split=0.2,
    subset="training",
    seed=123
)

# Load validation data
validation_data = tf.keras.utils.image_dataset_from_directory(
    train_path,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True,
    validation_split=0.2,
    subset="validation",
    seed=123
)

# Load test data
test_data = tf.keras.utils.image_dataset_from_directory(
    test_path,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

# Classes
class_names = train_data.class_names

print("Classes:")
print(class_names)

# Improve loading speed
AUTOTUNE = tf.data.AUTOTUNE

train_data = train_data.prefetch(AUTOTUNE)
validation_data = validation_data.prefetch(AUTOTUNE)
test_data = test_data.prefetch(AUTOTUNE)

# CNN model
model = models.Sequential([
    layers.Input(shape=(128, 128, 3)),
    layers.Rescaling(1./255),

    layers.Conv2D(32, 3, activation="relu"),
    layers.MaxPooling2D(),

    layers.Conv2D(64, 3, activation="relu"),
    layers.MaxPooling2D(),

    layers.Conv2D(128, 3, activation="relu"),
    layers.MaxPooling2D(),

    layers.Flatten(),

    layers.Dense(128, activation="relu"),
    layers.Dropout(0.5),

    layers.Dense(9, activation="softmax")
])

# Compile
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print("Model created successfully!")

# Train
history = model.fit(
    train_data,
    validation_data=validation_data,
    epochs=10
)

# Evaluate
test_loss, test_accuracy = model.evaluate(test_data)

print("Test Accuracy:", test_accuracy)

# Save model
model.save("models/student_engagement_cnn.keras")

print("Model saved successfully!")