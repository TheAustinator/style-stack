import datetime as dt
from itertools import product
from keras.applications import VGG16, vgg16
from matplotlib.backends.backend_pdf import PdfPages
import logging
import matplotlib.pyplot as plt
import numpy as np
import random
import os

from utils import get_concatenated_images, get_image_paths, plot_results, load_image
from gram_matrices import gram_matrix
from style_similarity_search import (generate_embeddings, get_closest_indices,
                                     get_embedding_model, pca)

logger = logging.getLogger()
logger.setLevel(logging.INFO)

def main():
    image_paths = get_image_paths('../data/raw')
    query_idx = int(len(image_paths) * random.random()) # TODO: change to np.random
    image_path = image_paths[query_idx]
    img, x = load_image(image_path)

    model = VGG16(weights='imagenet', include_top=True)
    embedding_list = [layer.output for layer in model.layers]
    for emb in embedding_list:
        style_loss =

    valid_image_paths, embeddings = generate_embeddings(embedding_model, image_paths)
    features = pca(embeddings, 300)
    query_image_idx = int(len(valid_image_paths) * random.random())
    closest_image_indices = get_closest_indices(query_image_idx, features)
    query_image = get_concatenated_images(valid_image_paths, [query_image_idx])
    results_image = get_concatenated_images(valid_image_paths, closest_image_indices)

    plot_results(query_image_idx, query_image, closest_image_indices, results_image)


if __name__ == '__main__':
    main()
