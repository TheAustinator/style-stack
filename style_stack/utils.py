import json
from keras.applications.imagenet_utils import preprocess_input
from keras.preprocessing import image
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


def get_image_paths(images_dir, max_num_images=10000):
    image_extensions = ['.jpg', '.png', '.jpeg']
    # TODO: shorten with glob
    # TODO: change to iterator instead of max_num_images
    image_paths = [os.path.join(dp, f) for dp, dn, filenames in
                   os.walk(images_dir) for f in filenames if
                   os.path.splitext(f)[1].lower() in image_extensions]

    if max_num_images < len(image_paths):
        image_paths = [image_paths[i] for i in sorted(random.sample(
            range(len(image_paths)), max_num_images))]
    print(f'keeping {len(image_paths)} image_paths to analyze')
    return image_paths


def get_concatenated_images(indexes, image_paths, thumb_height):
    thumbs = []
    for idx in indexes:
        img = image.load_img(image_paths[idx])
        img = img.resize((int(img.width * thumb_height / img.height), thumb_height))
        thumbs.append(img)
    concat_image = np.concatenate([np.asarray(t) for t in thumbs], axis=1)
    return concat_image


def plot_results(results_filename, results_dir='../output/'):
    with open(os.path.join(results_dir, results_filename)) as f:
        json_str = json.load(f)
        results = {str(k): v for k, v in json_str.items()}
    results_files = results['results_files'].values()
    model = results['model']
    similarity_weights = results['similarity_weights']
    lib_name = results['lib_name']
    n_images = results['n_images']
    query_img_path = results['query_img']

    query_img = image.load_img(query_img_path)

    plt.figure(figsize=(5, 5))
    plt.imshow(query_img)
    plt.title(f'query image')

    plt.figure(figsize=(16, 12))
    plt.imshow(results_img)
    plt.title(f'result images: {results_indices}')

