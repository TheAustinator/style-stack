"""
Implementation based on Andrew Look's work at Plato Designs
"""

from keras.applications import VGG16
from keras.applications.imagenet_utils import preprocess_input
from keras.utils import plot_model
from keras.models import Model
from keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np
import os
import random
from sklearn.decomposition import PCA
from scipy.spatial import distance

from utils import get_image_paths, load_image, get_concatenated_images, plot_results


def main():
    image_paths = get_image_paths('../data/raw')
    model = VGG16(weights='imagenet', include_top=True)
    layer_name, embedding_model_fc2 = get_embedding_model(model, 1)
    valid_image_paths, embeddings = generate_embeddings(embedding_model_fc2, image_paths)
    features = pca(embeddings, 300)
    query_image_idx = int(len(valid_image_paths) * random.random())
    closest_image_indices = get_closest_indices(query_image_idx, features)
    query_image = get_concatenated_images(valid_image_paths, [query_image_idx])
    results_image = get_concatenated_images(valid_image_paths, closest_image_indices)
    plot_results(query_image_idx, query_image, closest_image_indices, results_image)


def get_embedding_model(model, layer_idx_from_output):
    layer_idx = len(model.layers) - layer_idx_from_output - 1
    layer_name = model.layers[layer_idx].name
    embedding_model = Model(inputs=model.input,
                            outputs=model.get_layer(f'{layer_name}').output)
    return layer_name, embedding_model


def generate_embeddings(embedding_model, image_paths, log_failures=False):
    embeddings = []
    valid_image_paths = []
    invalid_image_paths = []
    for i, image_path in enumerate(image_paths):
        if i % 1000 == 0:
            print("analyzing image %d / %d" % (i, len(image_paths)))
        try:
            _, x = load_image(image_path, embedding_model.input_shape[1:3])
        except Exception as e:
            invalid_image_paths.append(image_path)
            continue
        else:
            emb = embedding_model.predict(x)[0]
            embeddings.append(emb)
            valid_image_paths.append(image_path)  # only keep ones that didnt cause errors

    # TODO: add logging for invalid_images
    print(f'finished extracting {len(embeddings)} embeddings '
          f'for {len(valid_image_paths)} images with {len(invalid_image_paths)} failures')
    return valid_image_paths, embeddings


def pca(embeddings, n_components):
    embeddings = np.array(embeddings)
    pca = PCA(n_components=n_components)
    pca.fit(embeddings)
    pca_features = pca.transform(embeddings)
    return pca_features


def get_closest_indices(query_image_idx, features, num_results=5, query_from_library=True):
    distances = [distance.cosine(features[query_image_idx], feat) for feat in features]
    start_idx = 1 if query_from_library else 0
    indices_closest = sorted(range(len(distances)), key=lambda k: distances[k])[
                      start_idx:start_idx + num_results + 1]
    return indices_closest


if __name__ == '__main__':
    main()
