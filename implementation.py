import tensorflow as tf
import numpy as np

loaded_model = tf.keras.models.load_model('model.h5')

desired_pos_orn = np.array([[0.5, 0,5, 0,5, 0, 0, 1]])

prediction = loaded_model.predict(desired_pos_orn)
print(f"Predicted class for the new sample: {prediction}")