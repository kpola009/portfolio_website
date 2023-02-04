import tensorflow as tf
import numpy as np

model = tf.keras.models.load_model("images/Projects Content/Pneumonia Classification/PNEUMONIA")

def classify_pneumonia(image_name: object):
    if image_name == 'IMAGE 1':
        image = "images/Projects Content/Pneumonia Classification/IM-0187-0001.jpeg"
        sample_image = tf.keras.utils.load_img(image, grayscale = True, target_size=(150,150))
        input_arr = tf.keras.utils.img_to_array(sample_image)
        input_arr = np.array([input_arr])
        prediction = model.predict(input_arr)
        if prediction <= 0.5:
            return(0)
        elif prediction > 0.5:
            return(1)


    elif image_name == 'IMAGE 2':
        image = "images/Projects Content/Pneumonia Classification/person1_bacteria_2.jpeg"
        sample_image = tf.keras.utils.load_img(image, grayscale=True, target_size=(150, 150))
        input_arr = tf.keras.utils.img_to_array(sample_image)
        input_arr = np.array([input_arr])
        prediction = model.predict(input_arr)
        if prediction <= 0.5:
            return (0)
        elif prediction > 0.5:
            return (1)

    elif image_name == 'IMAGE 3':
        image = "images/Projects Content/Pneumonia Classification/person2_bacteria_3.jpeg"
        sample_image = tf.keras.utils.load_img(image, grayscale=True, target_size=(150, 150))
        input_arr = tf.keras.utils.img_to_array(sample_image)
        input_arr = np.array([input_arr])
        prediction = model.predict(input_arr)
        if prediction <= 0.5:
            return (0)
        elif prediction > 0.5:
            return (1)

