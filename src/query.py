import datetime as dt
import faiss
import json
from keras.applications.vgg16 import VGG16
import numpy as np
import os

from utils import get_image_paths
from gram_matrices import build_gram_lib_index, build_query_gram_dict, gen_images_embeddings, gram_matrix, load_gram_lib, plot_results, query_gram_lib_index, save_gram_lib


def main():
    lib_name = 'test_1'
    query_idx = 0
    k = 4
    similarity_weights = {
        'block3_conv1': 1,
        'block3_conv2': 1,
        'block3_conv3': 0.5,
    }

    model = VGG16(weights='imagenet', include_top=True)
    image_paths = get_image_paths('../data/mini')
    query_path = image_paths[query_idx]
    images_embeddings, layer_names, file_mapping = gen_images_embeddings([query_path], model)
    query_layer_names = list(similarity_weights)

    file_mapping, index_dict = load_gram_lib(lib_name)

    # build_query_gram
    # TODO: change this to use image path
    img_embeddings = images_embeddings[query_idx]
    query_gram_dict = build_query_gram_dict(img_embeddings, layer_names)

    results_indices = query_gram_lib_index(query_gram_dict, layer_names, index_dict, similarity_weights, k)
    results_files = {i: file_mapping[i] for i in results_indices}
    results = {
        'model': model.name,
        'query_img': query_path,
        'lib_name': lib_name,
        'n_images': len(image_paths),
        'similarity_weights': similarity_weights,
        'results_files': results_files,
    }
    timestamp = str(dt.datetime.now())
    output_file = f'../output/results-{timestamp}'
    with open(output_file, 'w') as f:
        json.dump(results, f)


if __name__ == '__main__':
    main()
