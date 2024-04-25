from flask import Flask, jsonify, request
from flask_cors import CORS
import subprocess
import os
import time
from PIL import Image
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt
import base64
import io
from io import BytesIO
os.environ["TFHUB_DOWNLOAD_PROGRESS"] = "True"

app = Flask(__name__)
CORS(app)

SAVED_MODEL_PATH = "https://tfhub.dev/captain-pool/esrgan-tf2/1"

def preprocess_image(image_data):
  """ Loads image from path and preprocesses to make it model ready
      Args:
        image_path: Path to the image file
  """
  hr_image = tf.image.decode_image(image_data)
  # If PNG, remove the alpha channel. The model only supports
  # images with 3 color channels.
  if hr_image.shape[-1] == 4:
    hr_image = hr_image[...,:-1]
  hr_size = (tf.convert_to_tensor(hr_image.shape[:-1]) // 4) * 4
  hr_image = tf.image.crop_to_bounding_box(hr_image, 0, 0, hr_size[0], hr_size[1])
  hr_image = tf.cast(hr_image, tf.float32)
  return tf.expand_dims(hr_image, 0)

def save_image(image, filename):
  """
    Saves unscaled Tensor Images.
    Args:
      image: 3D image tensor. [height, width, channels]
      filename: Name of the file to save.
  """
  if not isinstance(image, Image.Image):
    image = tf.clip_by_value(image, 0, 255)
    image = Image.fromarray(tf.cast(image, tf.uint8).numpy())
#   image.save("%s.jpg" % filename)
#   print("Saved as %s.jpg" % filename)
  return image

def plot_image(image, title=""):
  """
    Plots images from image tensors.
    Args:
      image: 3D image tensor. [height, width, channels].
      title: Title to display in the plot.
  """
  image = np.asarray(image)
  image = tf.clip_by_value(image, 0, 255)
  image = Image.fromarray(tf.cast(image, tf.uint8).numpy())
  plt.imshow(image)
  plt.axis("off")
  plt.title(title)

@app.route('/img', methods=['POST'])
def get_img():
    img = request.json['img']
    
    img_d = base64.b64decode(img)
    
    img_w = base64.urlsafe_b64encode(img_d)
    
    img_b = tf.io.decode_base64(img_w)
    
    hr_image = preprocess_image(img_b)
    
    # Plotting Original Resolution image
    # plot_image(tf.squeeze(hr_image), title="Original Image")
    # save_image(tf.squeeze(hr_image), filename="Original Image")
    
    model = hub.load(SAVED_MODEL_PATH)
    
    fake_image = model(hr_image)
    fake_image = tf.squeeze(fake_image)
    
    # Plotting Super Resolution Image
    plot_image(tf.squeeze(fake_image), title="Super Resolution")
    s_img = save_image(tf.squeeze(fake_image), filename="Super Resolution")
    
    buffered = BytesIO()
    s_img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue())
    
    # print(img_str)
    return jsonify({'img': 'success', 'data' : img_str.decode('ascii')})
@app.route('/date', methods=['GET'])
def get_date():
    return jsonify({'date': '2020-01-01'})

if __name__ == '__main__':
    app.run()