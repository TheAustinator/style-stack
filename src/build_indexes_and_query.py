import datetime as dt
import faiss
import json
from keras.applications.vgg16 import VGG16
import numpy as np
import os
from itertools import product


from utils import get_image_paths
from gram_matrices import build_gram_lib_index, build_query_gram_dict, gen_images_embeddings, gram_matrix, load_gram_lib, plot_results, query_gram_lib_index, save_gram_lib
from query import main

def build_indexes_and_query(query_path, similarity_weights, k):
    lib_name = 'test_1'

    model = VGG16(weights='imagenet', include_top=True)
    image_paths = get_image_paths('../data/raw')
    images_embeddings, layer_names, file_mapping = gen_images_embeddings(image_paths, model)

    for layer_name in layer_names:
        if layer_name not in similarity_weights:
            similarity_weights.update({layer_name: 0})


    # create library
    index_dict = build_gram_lib_index(images_embeddings, layer_names, buffer_size=100)

    save_gram_lib(index_dict, image_paths, lib_name)

    # build_query_gram
    # TODO: change this to use image path
    img_embeddings, _, _ = gen_images_embeddings([query_path], model)
    img_embeddings = img_embeddings[0]
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
    k = 10

    # image_1 = '../data/test/72027-Coupons-general__dribbble-1304247178_teaser.png'
    # image_2 = '../data/test/108537-DJ-Controller-iPad-app__shot_1296582330.jpg'
    # image_3 = '../data/test/127434-iPhone-App-UI__shot_1299698639.jpg'
    # image_4 = '../data/test/134868-TRON-Mobile-ver__shot_1300800072.jpg'
    # image_5 = '../data/test/142362-Louis-Vuitton-Signature-Collection__pean.jpg'

    image_1 = '1412803-OS-X-App-Icons__icons.png'
    image_2 = '1069017-GoodNotes-icon-in-process__goodnotes.png'
    image_3 = '1230440-B-Dog__b_dog.jpg'
    image_4 = '3309097-Remasi-Webpage__remasi_webpage_green.jpg'
    image_5 = '4863170-Panda-sketch__panda.png'

    weights_block_1 = {
        'block1_conv1': 1,
        'block1_conv2': 1,
    }
    weights_block_2 = {
        'block2_conv1': 1,
        'block2_conv2': 1,
    }
    weights_block_3 = {
        'block3_conv1': 1,
        'block3_conv2': 1,
        'block3_conv3': 1,
    }
    weights_block_4 = {
        'block4_conv1': 1,
        'block4_conv2': 1,
        'block4_conv3': 1,
    }
    weights_block_5 = {
        'block5_conv1': 1,
        'block5_conv2': 1,
        'block5_conv3': 1,
    }

    weights_all_conv = {
        'block1_conv1': 1,
        'block1_conv2': 1,
        'block2_conv1': 1,
        'block2_conv2': 1,
        'block3_conv1': 1,
        'block3_conv2': 1,
        'block3_conv3': 1,
        'block4_conv1': 1,
        'block4_conv2': 1,
        'block4_conv3': 1,
    }

    weights_all_pool = {
        'block1_pool': 1,
        'block2_pool': 1,
        'block3_pool': 1,
        'block4_pool': 1,
        'block5_pool': 1,
    }

    query_images = [image_1, image_2, image_3, image_4, image_5]
    query_weights = [
        weights_block_1, weights_block_2, weights_block_3, weights_block_4,
        weights_block_5, weights_all_conv, weights_all_pool,
    ]
    search_space = list(product(query_images, query_weights))

    build_indexes_and_query(image_1, weights_block_1, k)
    for params in search_space:
        main(*params, k)
