import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split





np.random.seed(42)

num_samples = 1000
ages = np.random.randint(18, 70, size=num_samples)
bmis = np.random.uniform(18.5, 40, size=num_samples)
blood_sugar_levels = np.random.uniform(70, 140, size=num_samples)


X = np.vstack((ages, bmis, blood_sugar_levels)).T

y = np.logical_and(bmis > 25, blood_sugar_levels > 100).astype(int)


def minmaxscale(X):
    return (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))

X_scaled = minmaxscale(X)


X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)



model = Sequential([
    Dense(10, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(8, activation='relu'),
    Dense(8, activation='relu'),
    Dense(4, activation='relu'),
    Dense(1, activation='sigmoid')
])


model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.summary()


# Training the model
model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2, verbose=1)

loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss}, Test Accuracy: {accuracy}")