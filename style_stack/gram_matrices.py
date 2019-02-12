import datetime as dt
import faiss
import glob
import json
from keras.applications.imagenet_utils import preprocess_input
import keras.backend as K
from keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np
import os
import re


def load_image(path, target_size):
    # TODO: compare to vgg19.preprocess input
    img = image.load_img(path, target_size=target_size)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return img, x





def gram_matrix(x):
    if np.ndim(x) == 4 and x.shape[0] == 1:
        x = x[0, :]
    elif np.ndim != 3:
        # TODO: make my own error
        raise ValueError(f'')
    x = x.reshape(x.shape[-1], -1)
    gram = np.dot(x, np.transpose(x))
    return gram

def test_inputs(query_idx, k, similarity_weights):
    pass

def _legacy_build_gram_lib(images_embeddings, layer_list):
    """

    :param images_embeddings:
    :param layer_list:
    :return:
    """
    n_embedding_layers = len(layer_list)
    # TODO: make this gram_list_dict
    gram_list_list = [[] for _ in range(n_embedding_layers)]
    for i, img_embeddings in enumerate(images_embeddings):
        for k, emb in enumerate(img_embeddings):
            gram = gram_matrix(emb)
            gram_list_list[k].append(gram)

    gram_stack_dict = {}
    for i, gram_list in enumerate(gram_list_list):
        gram_stack = np.stack(gram_list)
        gram_stack_dict.update({layer_list[i]: gram_stack})

    return gram_stack_dict


def build_query_gram_dict(img_embeddings, layer_list):
    gram_dict = {}
    for i, emb in enumerate(img_embeddings):
        gram = gram_matrix(emb)
        gram_flat = gram.flatten()
        gram_exp = np.expand_dims(gram_flat, axis=0)
        gram_dict[layer_list[i]] = gram_exp
    return gram_dict


def build_gram_lib_index(images_embeddings, layer_names, buffer_size):
    start = dt.datetime.now()
    index_dict = {}
    gram_list_list = [[] for _ in range(len(layer_names))]

    def _index_buffer():
        """
        Helper method to move data from buffer to index when `buffer_size` is
        reached
        """
        nonlocal gram_list_list
        nonlocal index_dict

        for j, gram_list in enumerate(gram_list_list):
            gram_stack = np.stack(gram_list)
            index_dict[layer_names[j]].add(gram_stack)
            gram_list_list = [[] for _ in range(len(gram_list_list))]

    for i, img_embeddings in enumerate(images_embeddings):

        for k, emb in enumerate(img_embeddings):
            gram = gram_matrix(emb)
            gram_flat = gram.flatten()
            gram_list_list[k].append(gram_flat)

            if i == 0:
                d = len(gram_flat)
                index_dict[f'{layer_names[k]}'] = faiss.IndexFlatL2(d)

        if i % buffer_size == 0 and i > 0:
            _index_buffer()

    if gram_list_list:
        _index_buffer()
    end = dt.datetime.now()
    index_time = (end - start).microseconds / 1000
    print(f'index time: {index_time} ms')
    return index_dict


def save_gram_lib(index_dict, file_mapping, lib_name):
    output_dir = f'../data/indexes/{lib_name}/'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for layer_name, index in index_dict.items():
        filename = f'grams-{layer_name}.index'
        filepath = os.path.join(output_dir, filename)
        faiss.write_index(index, filepath)

    filename = 'file_mapping.json'
    filepath = os.path.join(output_dir, filename)
    with open(filepath, 'w') as f:
        json.dump(file_mapping, f)


def load_gram_lib(lib_name):
    input_dir = f'../data/indexes/{lib_name}/'
    with open(os.path.join(input_dir, 'file_mapping.json')) as f:
        index_str = json.load(f)
        file_mapping = {int(k): str(v) for k, v in index_str.items()}

    index_dict = {}

    gram_layer_files = glob.glob(os.path.join(input_dir, 'grams-*.index'))
    for f in gram_layer_files:
        layer_name = re.search(f'{input_dir}grams-(.+?)\.index', f).group(1)
        index = faiss.read_index(f)
        index_dict.update({layer_name: index})

    return file_mapping, index_dict


def query_gram_lib_index(query_gram_dict, layer_names, index_dict, similarity_weights, n_results):
    start = dt.datetime.now()
    proximal_indices = set()
    for layer_name, gram in query_gram_dict.items():
        _, closest_indices = index_dict[layer_name].search(gram, n_results)
        proximal_indices.update(closest_indices[0].tolist())

    dist_dict = {}
    for layer_name, gram in query_gram_dict.items():

        labels_iter_range = list(range(1, len(proximal_indices) + 1))
        labels = np.array([list(proximal_indices), labels_iter_range])
        distances = np.empty((1, len(proximal_indices)), dtype='float32')
        index_dict[layer_name].compute_distance_subset(1, faiss.swig_ptr(gram), len(proximal_indices), faiss.swig_ptr(distances), faiss.swig_ptr(labels))
        distances = distances.flatten()
        norm_distances = distances / max(distances)
        dist_dict[layer_name] = {idx: norm_distances[i] for i, idx in enumerate(proximal_indices)}

    print(dist_dict)

    weighted_dist_dict = {}
    for idx in proximal_indices:
        weighted_dist = sum([similarity_weights[layer_name] * dist_dict[layer_name][idx] for layer_name in similarity_weights])

        weighted_dist_dict[idx] = weighted_dist

    print(weighted_dist_dict)

    indices = sorted(weighted_dist_dict, key=weighted_dist_dict.get)
    results_indices = indices[:n_results]

    end = dt.datetime.now()
    index_time = (end - start).microseconds / 1000
    print(f'query time: {index_time} ms')
    print(results_indices)
    return results_indices


def _legacy_save_gram_lib(lib_name, gram_lib, file_mapping):
    for layer_name, gram_stack in gram_lib.items():
        np.save(f'{lib_name}-grams-{layer_name}.npy', gram_stack)
        with open(f'{lib_name}-file_mapping.json', 'w') as f:
            json.dump(file_mapping, f)


def _legacy_load_gram_lib(lib_name):
    with open(f'{lib_name}-file_mapping.json') as f:
        index_str = json.load(f)
        file_index = {int(k): str(v) for k, v in index_str.items()}

    gram_lib = {}
    gram_layer_files = sorted(glob.glob(f'{lib_name}-grams-*.npy'))
    for f in gram_layer_files:
        layer_name = re.search(f'{lib_name}-grams-(.+?)\.npy', f).group(1)
        gram_stack = np.load(f)
        gram_lib.update({layer_name: gram_stack})

    return file_index, gram_lib


def gen_mock_grams():
    gram_lib = [np.ones((7, i, i)) * i for i in range(2, 6)]
    query_grams = [np.ones((i, i)) * 100 * i for i in range(2, 6)]
    return query_grams, gram_lib


def calc_layer_sum_sq(query_gram, gram_lib):
    gram_sum_sq = np.sum(np.square(gram_lib - query_gram), axis=(1, 2)) / (query_gram.size)
    return gram_sum_sq


def query_gram_lib(query_gram_dict, gram_lib, layer_names, n_results=3):
    sum_sq_by_layer = []
    for layer_name in layer_names:
        layer_query_gram = query_gram_dict[layer_name]
        layer_lib_grams = gram_lib[layer_name]
        sum_sq_arr = calc_layer_sum_sq(layer_query_gram, layer_lib_grams)
        sum_sq_by_layer.append(sum_sq_arr)

    sum_sq_tot = sum(sum_sq_by_layer)

    if not len(sum_sq_by_layer) == len(layer_names):
        raise ValueError()
    if not len(sum_sq_tot) == gram_lib[layer_names[0]].shape[0]:
        raise ValueError()

    results_indices = sum_sq_tot.argsort()[:n_results]
    return results_indices


def get_concatenated_images(indexes, image_paths, thumb_height):
    thumbs = []
    for idx in indexes:
        img = image.load_img(image_paths[idx])
        img = img.resize((int(img.width * thumb_height / img.height), thumb_height))
        thumbs.append(img)
    concat_image = np.concatenate([np.asarray(t) for t in thumbs], axis=1)
    return concat_image


def plot_results(query_idx, results_indices, image_paths):
    query_img = get_concatenated_images([query_idx], image_paths, 200)
    results_img = get_concatenated_images(results_indices, image_paths, 200)

    # display the query image
    plt.figure(figsize=(5, 5))
    plt.imshow(query_img)
    plt.title(f'query image {query_idx}')

    # display the resulting images
    plt.figure(figsize=(16, 12))
    plt.imshow(results_img)
    plt.title(f'result images')
