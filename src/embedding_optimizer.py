import datetime as dt
from itertools import product
from keras.applications import VGG16
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import numpy as np
import random

from style_similarity_search import generate_embeddings, get_closest_indices,\
    get_embedding_model, pca
from utils import get_concatenated_images, get_image_paths


def main():
    image_paths = get_image_paths('../data/raw')
    model = VGG16(weights='imagenet', include_top=True)

    query_indices = [int(len(image_paths) * random.random()) for _ in range(5)]
    embedding_layer_space = range(len(model.layers))
    pca_dim_space = np.geomspace(32, 2048, num=7)
    search_space = list(product(query_indices, embedding_layer_space, pca_dim_space))

    timestamp = dt.datetime.now()
    pdf_pages = PdfPages(f'../output/embedding_selection_{timestamp}.pdf')
    for params in search_space:
        query_idx, embedding_layer_idx, pca_dim = params
        _, embedding_model = get_embedding_model(model, embedding_layer_idx)
        valid_image_paths, embeddings = generate_embeddings(embedding_model, image_paths)
        features = pca(embeddings, 300)
        query_image_idx = int(len(valid_image_paths) * random.random())
        closest_image_indices = get_closest_indices(query_image_idx, features)
        query_image = get_concatenated_images(valid_image_paths, [query_image_idx])
        results_image = get_concatenated_images(valid_image_paths, closest_image_indices)

        fig = plt.figure(figsize=(8.27, 11.69), dpi=100)
        plt.subplot2grid((2, 1), (0, 0))
        plt.imshow(query_image)
        plt.title(f'query image {query_image_idx}')
        plt.subplot2grid((2, 1), (1, 0))
        plt.imshow(results_image)
        plt.title(f'result images: {closest_image_indices}')
        pdf_pages.savefig(fig)
    pdf_pages.close()


if __name__ == '__main__':
    main()
