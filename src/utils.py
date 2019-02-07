
from keras.applications.imagenet_utils import preprocess_input
from keras.preprocessing import image

from keras.applications import vgg16
import matplotlib.pyplot as plt
import numpy as np
import os
import random


def load_image(path, target_size):
    # TODO: compare to vgg19.preprocess input
    img = image.load_img(path, target_size=target_size)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return img, x


def get_image_paths(images_dir):
    image_extensions = ['.jpg', '.png', '.jpeg']
    max_num_images = 10000
    # TODO: shorten with glob
    image_paths = [os.path.join(dp, f) for dp, dn, filenames in
                   os.walk(images_dir) for f in filenames if
                   os.path.splitext(f)[1].lower() in image_extensions]
    # image_paths = [
    #     image_path
    #     for image_path in random.choice(image_paths), max_num_images, replace = False))
    # ]
    if max_num_images < len(image_paths):
        image_paths = [image_paths[i] for i in sorted(random.sample(
            range(len(image_paths)), max_num_images))]
    print(f'keeping {len(image_paths)} image_paths to analyze')
    return image_paths


def get_concatenated_images(valid_imgs, indexes, thumb_height=100):
    thumbs = []
    for idx in indexes:
        img = image.load_img(valid_imgs[idx])
        img = img.resize((int(img.width * thumb_height / img.height), thumb_height))
        thumbs.append(img)
    concat_image = np.concatenate([np.asarray(t) for t in thumbs], axis=1)
    return concat_image


def plot_results(query_img_idx, query_img, results_indices, results_img):
    plt.figure(figsize=(5, 5))
    plt.imshow(query_img)
    plt.title(f'query image {query_img_idx}')

    plt.figure(figsize=(16, 12))
    plt.imshow(results_img)
    plt.title(f'result images: {results_indices}')

