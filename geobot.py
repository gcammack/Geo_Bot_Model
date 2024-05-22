

import tensorflow as tf
import numpy as np

from tensorflow.keras.preprocessing.image import ImageDataGenerator

import matplotlib.pyplot as plt

dataset_dir = "/work/cse479/jseibel/downsample"


batch_size = 32
image_size = (128, 128)
class_names = [str(i) for i in range(1,29 )]
NUM_CLASSES = len(class_names)



train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    dataset_dir,
    validation_split=0.2,
    label_mode='categorical',
    subset="training",
    seed=422323,
    image_size=image_size,
    batch_size=batch_size,
    class_names=class_names
)

validation_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    dataset_dir,
    validation_split=0.2,
    label_mode='categorical',
    subset="validation",
    seed=422323,
    image_size=image_size,
    batch_size=batch_size,
    class_names=class_names
)





# Visualize a few examples
plt.figure(figsize=(10, 10))
for images, labels in train_dataset.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(train_dataset.class_names[tf.argmax(labels[i])])  
        plt.axis("off")
plt.show()

def read_csv_to_dict(file_path):
    data_dict = {}

    with open(file_path, 'r') as file:
        for line in file:
            label, lat, lon = line.strip().split(',')

            lat = float(lat)
            lon = float(lon)

            data_dict[label] = {'latitude': lat, 'longitude': lon}

    return data_dict



from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import regularizers



input_shape = (128, 128, 3)

num_classes = 28


zone_center_file = './centersd.txt'
geocell_centers_dict = read_csv_to_dict(zone_center_file)

# Convert geocell_centers_dict to tensors
geocell_centers = tf.constant([[coords["latitude"], coords["longitude"]] for coords in geocell_centers_dict.values()])

def custom_loss(y_true, y_pred, geocell_centers):
    ce_loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred, from_logits=False)

    spatial_penalty = calculate_spatial_penalty(y_true, y_pred, geocell_centers)

    weighted_loss = tf.reduce_mean(ce_loss * spatial_penalty)

    return weighted_loss

def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points
    on the earth specified in decimal degrees.
    """
    # Convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    r = 3956
    return c * r

def calculate_spatial_penalty(y_true, y_pred, geocell_centers):
    true_lat = tf.gather(geocell_centers[:, 0], tf.argmax(y_true, axis=1))
    true_lon = tf.gather(geocell_centers[:, 1], tf.argmax(y_true, axis=1))
    pred_lat = tf.gather(geocell_centers[:, 0], tf.argmax(y_pred, axis=1))
    pred_lon = tf.gather(geocell_centers[:, 1], tf.argmax(y_pred, axis=1))

    true_lat_rad = tf.math.scalar_mul(tf.constant(np.pi / 180.0, dtype=tf.float32), true_lat)
    true_lon_rad = tf.math.scalar_mul(tf.constant(np.pi / 180.0, dtype=tf.float32), true_lon)
    pred_lat_rad = tf.math.scalar_mul(tf.constant(np.pi / 180.0, dtype=tf.float32), pred_lat)
    pred_lon_rad = tf.math.scalar_mul(tf.constant(np.pi / 180.0, dtype=tf.float32), pred_lon)

    # Calculate Haversine distance
    dlat = tf.math.sin(tf.math.subtract(pred_lat_rad, true_lat_rad) / 2)
    dlon = tf.math.sin(tf.math.subtract(pred_lon_rad, true_lon_rad) / 2)
    a = tf.math.add(tf.math.multiply(dlat, dlat),
                    tf.math.multiply(tf.math.multiply(tf.math.cos(true_lat_rad),
                                                      tf.math.cos(pred_lat_rad)),
                                     tf.math.multiply(dlon, dlon)))
    c = 2 * tf.math.atan2(tf.math.sqrt(a), tf.math.sqrt(1 - a))
    distance = tf.math.scalar_mul(tf.constant(3956, dtype=tf.float32), c)  


    return distance


loss_fn = lambda y_true, y_pred: custom_loss(y_true, y_pred, geocell_centers)


from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping

early_stopping = EarlyStopping(monitor='val_loss', patience=15)


conv_base = MobileNetV2(weights='imagenet',
                        include_top=False,
                        input_shape=input_shape)

# Unlock the last two convolutional layers
for layer in conv_base.layers[:-3]:
    layer.trainable = True

model = models.Sequential([
    conv_base,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.1)),
    layers.BatchNormalization(),
    layers.Dropout(0.75),
    layers.Dense(num_classes, activation='softmax')
])


model.compile(optimizer=optimizers.Adam(learning_rate=0.0002),
              loss= loss_fn,
              metrics=['accuracy', 'top_k_categorical_accuracy'])


model.summary()

history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=100,
    callbacks=[early_stopping]
)



model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.1), input_shape=input_shape),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.1)),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.1)),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(256, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.1)),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(256, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.1)),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.1)),
    layers.Dropout(0.8),
    layers.BatchNormalization(),
    layers.Dense(num_classes, activation='softmax')
])

model.compile(optimizer=optimizers.Adam(learning_rate=0.0002),
              loss=loss_fn,
              metrics=['accuracy', 'top_k_categorical_accuracy'])

model.summary()
history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=100,
)



from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras import Model
from PIL import Image

model = load_model('coolmodel.keras', custom_objects={'<lambda>': loss_fn})
image_path = 'loc1.jpg'
image = tf.io.read_file(image_path)
image = tf.image.decode_image(image, channels=3) 
target_size = (196, 196)  
resized_image = tf.image.resize(image, target_size)


def grad_cam_plus(model, img,
                  layer_name="block5_conv3", label_name=None,
                  category_id=None):
    """Get a heatmap by Grad-CAM++.

    Args:
        model: A model object, build from tf.keras 2.X.
        img: An image ndarray.
        layer_name: A string, layer name in model.
        label_name: A list or None,
            show the label name by assign this argument,
            it should be a list of all label names.
        category_id: An integer, index of the class.
            Default is the category with the highest score in the prediction.

    Return:
        A heatmap ndarray(without color).
    """
    img_tensor = np.expand_dims(img, axis=0)

    conv_layer = model.get_layer(layer_name)
    heatmap_model = Model([model.inputs], [conv_layer.output, model.output])

    with tf.GradientTape() as gtape1:
        with tf.GradientTape() as gtape2:
            with tf.GradientTape() as gtape3:
                conv_output, predictions = heatmap_model(img_tensor)
                if category_id is None:
                    category_id = np.argmax(predictions[0])
                if label_name is not None:
                    print(label_name[category_id])
                output = predictions[:, category_id]
                conv_first_grad = gtape3.gradient(output, conv_output)
            conv_second_grad = gtape2.gradient(conv_first_grad, conv_output)
        conv_third_grad = gtape1.gradient(conv_second_grad, conv_output)

    global_sum = np.sum(conv_output, axis=(0, 1, 2))

    alpha_num = conv_second_grad[0]
    alpha_denom = conv_second_grad[0]*2.0 + conv_third_grad[0]*global_sum
    alpha_denom = np.where(alpha_denom != 0.0, alpha_denom, 1e-10)

    alphas = alpha_num/alpha_denom
    alpha_normalization_constant = np.sum(alphas, axis=(0,1))
    alphas /= alpha_normalization_constant

    weights = np.maximum(conv_first_grad[0], 0.0)

    deep_linearization_weights = np.sum(weights*alphas, axis=(0,1))
    grad_cam_map = np.sum(deep_linearization_weights*conv_output[0], axis=2)

    heatmap = np.maximum(grad_cam_map, 0)
    max_heat = np.max(heatmap)
    if max_heat == 0:
        max_heat = 1e-10
    heatmap /= max_heat

    return heatmap
heat_map = grad_cam_plus(model, resized_image, layer_name = 'conv2d_15' )

def show_imgwithheat(img_path, heatmap, alpha=0.3, return_array=False):
    """Show the image with heatmap.

    Args:
        img_path: string.
        heatmap: image array, get it by calling grad_cam().
        alpha: float, transparency of heatmap.
        return_array: bool, return a superimposed image array or not.
    Return:
        None or image array.
    """
    img = cv2.imread(img_path)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = (heatmap*255).astype("uint8")
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = heatmap * alpha + img
    superimposed_img = np.clip(superimposed_img, 0, 255).astype("uint8")
    superimposed_img = cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB)

    imgwithheat = Image.fromarray(superimposed_img)
    try:
        display(imgwithheat)
    except NameError:
        imgwithheat.show()

    if return_array:
        return superimposed_img

import os

directory = '/work/cse479/jseibel/downsample/20/'

# Loop through the files in the directory for gradcam vis
i = 0
for filename in os.listdir(directory):
    if os.path.isfile(os.path.join(directory, filename)):
        fp = directory + filename
        image = tf.io.read_file(fp)
        image = tf.image.decode_image(image, channels=3) 
        target_size = (196, 196)  
        resized_image = tf.image.resize(image, target_size)
        heat_map = grad_cam_plus(model, resized_image, layer_name = 'conv2d_16' )
        show_imgwithheat(fp, heat_map)

from sklearn.metrics import confusion_matrix

def extract_labels_and_predictions(dataset):
    true_labels = []
    predictions = []
    for images, labels in dataset:
        true_labels.extend(np.argmax(labels.numpy(), axis=1))
        predictions.extend(np.argmax(model.predict(images), axis=1))
    return true_labels, predictions

val_true_labels, val_predictions = extract_labels_and_predictions(validation_dataset)
val_cm = confusion_matrix(val_true_labels, val_predictions)
val_cm_normalized = val_cm.astype('float') / val_cm.sum(axis=1)[:, np.newaxis]


# Plot normalized confusion matrix
plt.figure(figsize=(10, 8))
plt.imshow(val_cm_normalized, interpolation='nearest', cmap=plt.cm.Blues)

plt.title('Normalized Test Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(num_classes)
plt.xticks(tick_marks, range(num_classes))
plt.yticks(tick_marks, range(num_classes))
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

import matplotlib.pyplot as plt

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

import matplotlib.pyplot as plt
plt.plot(train_top_k_accuracy)
plt.plot(val_top_k_accuracy)
plt.title('Model Top K Accuracy')
plt.ylabel('Top K Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

from keras.preprocessing import image


#Making predictions for game demo
image_path = 'loc5.jpg'
img = tf.io.read_file(image_path)
img = tf.image.decode_image(img, channels=3)  

resized_img = tf.image.resize(img, [196, 196])

img_array = image.img_to_array(resized_img)
img_array = np.expand_dims(img_array, axis=0)  

prediction = model.predict(img_array)

plt.imshow(resized_img.numpy().astype("uint8"))
plt.axis('off')
plt.show()

print(prediction)
print("Predicted class:", np.argmax(prediction))

#Calculate Distance
import math

def haversine_distance(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points
    on the earth specified in decimal degrees.
    """
    lon1 = math.radians(lon1)
    lat1 = math.radians(lat1)
    lon2 = math.radians(lon2)
    lat2 = math.radians(lat2)

    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    distance = 3956 * c  # Radius of earth miles
    return distance

def compute_distance(image_path, predicted_class):
    image_filename = image_path.split('/')[-1]
    longitude, latitude = map(float, image_filename.split(',')[:2])

    predicted_coordinates = geocell_centers[predicted_class]

    distance = haversine_distance(longitude, latitude, *predicted_coordinates)
    return distance


testdir = '/work/cse479/jseibel/downsample2'
dist = 0
for i in range(1, 29):
    test_class_dir = testdir + f'/{i}'
    for filename in os.listdir(test_class_dir):
        img = tf.io.read_file(test_class_dir + '/' +  filename)
        img = tf.image.decode_image(img, channels=3)
        resized_img = tf.image.resize(img, [196, 196])
        img_array = image.img_to_array(resized_img)
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array)
        prediction = np.argmax(prediction)
        dist += compute_distance(filename, prediction)

print(dist)

