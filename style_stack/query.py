import datetime as dt
import faiss
import json
from keras.applications.vgg16 import VGG16
import numpy as np
import os

from utils import get_image_paths
from gram_matrices import build_gram_lib_index, build_query_gram_dict, gen_images_embeddings, gram_matrix, load_gram_lib, plot_results, query_gram_lib_index, save_gram_lib


def main(query_path, similarity_weights, k):
    lib_name = 'test_1'

    model = VGG16(weights='imagenet', include_top=True)
    images_embeddings, layer_names, _ = gen_images_embeddings([query_path], model)
    query_layer_names = list(similarity_weights)

    file_mapping, index_dict = load_gram_lib(lib_name)

    # build_query_gram
    # TODO: change this to use image path
    img_embeddings = images_embeddings[0]
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
