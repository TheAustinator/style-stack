import glob
from keras.preprocessing import image
import keras.backend as K
import numpy as np
from keras.applications.imagenet_utils import preprocess_input
import matplotlib.pyplot as plt
import json
import re


def load_image(path, target_size):
    # TODO: compare to vgg19.preprocess input
    img = image.load_img(path, target_size=target_size)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return img, x


def gen_images_embeddings(image_paths, model):
    file_mapping = {i: f for i, f in enumerate(image_paths)}
    inputs = [load_image(path, model.input_shape[1:3])[1] for path in image_paths]

    conv_layers = model.layers[1:-4]
    layer_names = [layer.name for layer in conv_layers]
    embedding_layers = [layer.output for layer in conv_layers]
    functor = K.function([model.input], embedding_layers)
    images_embeddings = [functor([img, 1]) for img in inputs]
    return images_embeddings, layer_names, file_mapping


def gram_matrix(x):
    if np.ndim(x) == 4 and x.shape[0] == 1:
        x = x[0, :]
    elif np.ndim != 3:
        # TODO: make my own error
        raise ValueError(f'')
    x = x.reshape(x.shape[-1], -1)
    gram = np.dot(x, np.transpose(x))
    return gram


def build_gram_lib(images_embeddings, layer_list):
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


def build_gram_indices(lib_name, layer_list, d_list):
    pass


def add_to_gram_indices(lib_name, layer_list, images_embeddings):
    pass


def search_gram_indices():
    pass


def save_gram_lib(lib_name, gram_lib, file_mapping):
    for layer_name, gram_stack in gram_lib.items():
        np.save(f'{lib_name}-grams-{layer_name}.npy', gram_stack)
        with open(f'{lib_name}-file_mapping.json', 'w') as f:
            json.dump(file_mapping, f)


def load_gram_lib(lib_name):
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
    plt.figure(figsize=(5,5))
    plt.imshow(query_img)
    plt.title(f'query image {query_idx}')

    # display the resulting images
    plt.figure(figsize=(16,12))
    plt.imshow(results_img)
    plt.title(f'result images')
