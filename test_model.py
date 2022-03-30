import tensorflow as tf

from tf_frodo.frodo import FRODO

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

x_train = tf.expand_dims(x_train, axis=-1)
x_test = tf.expand_dims(x_test, axis=-1)

conv_model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, [7, 7]),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.ReLU(),
    tf.keras.layers.Conv2D(64, [3, 3], strides=[2, 2]),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.ReLU(),
    tf.keras.layers.Conv2D(128, [3, 3]),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.ReLU(),
    tf.keras.layers.GlobalAveragePooling2D(),
])

model = tf.keras.models.Sequential([
    conv_model,
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10)
])

predictions = model(x_train[:1]).numpy()

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

loss_fn(y_train[:1], predictions).numpy()

model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=1)


probability_model = tf.keras.Sequential([
    model,
    tf.keras.layers.Softmax()
])

probability_model.compile()
probability_model.predict(x_test)

f = FRODO(probability_model)

model_with_frodo = f.fit(x_test)

results = model_with_frodo.predict(x_test)
print(results)



