import datetime as dt
import faiss
import glob
import json
from keras.applications.vgg16 import VGG16
import keras.backend as K
import numpy as np
import os
import random
import re

from utils import load_image, get_image_paths


class GramStack:
    def __init__(self, *args):
        pass

    @classmethod
    def build(cls, image_dir, model, layer_range):
        inst = cls()
        inst.lib_name = None
        image_paths = get_image_paths(image_dir)
        inst.model = model
        inst.layer_range = layer_range
        inst._build_image_embedder([1, -4])
        lib_embeddings = inst._embed_library(image_paths)
        inst._build_index(lib_embeddings, buffer_size=1000)
        return inst

    @classmethod
    def load(cls, lib_name, layer_range):
        input_dir = f'../data/indexes/{lib_name}/'
        inst = cls()
        cls.lib_name = lib_name

        # all images must have loaded successfully at index build
        inst.invalid_paths = None

        # load metadata
        with open(os.path.join(input_dir, 'meta.json')) as f:
            json_str = json.load(f)
            metadata = {str(k): v for k, v in json_str.items()}
            if metadata['model'].lower() == 'vgg16':
                inst.model = VGG16(weights='imagenet', include_top=True)

        # load file mapping
        with open(os.path.join(input_dir, 'file_mapping.json')) as f:
            json_str = json.load(f)
            inst.file_mapping = {int(k): str(v) for k, v in json_str.items()}


        # load gram matrix indexes
        gram_layer_files = glob.glob(os.path.join(input_dir, 'grams-*.index'))
        inst.index_dict = {}
        for f in gram_layer_files:
            index = faiss.read_index(f)
            layer_name = re.search(f'{input_dir}grams-(.+?)\.index', f).group(1)
            inst.index_dict.update({layer_name: index})

        # build embedder from model
        inst._build_image_embedder(layer_range)

        return inst

    def add(self, image_dir):
        pass

    def save(self, lib_name):
        self.lib_name = lib_name
        output_dir = f'../data/indexes/{lib_name}/'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        for layer_name, index in self.index_dict.items():
            filename = f'grams-{layer_name}.index'
            filepath = os.path.join(output_dir, filename)
            faiss.write_index(index, filepath)

        mapping_path = os.path.join(output_dir, 'file_mapping.json')
        with open(mapping_path, 'w') as f:
            json.dump(self.file_mapping, f)

        metadata = {
            'model': self.model.name,
        }
        metadata_path = os.path.join(output_dir, 'meta.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f)

    def query(self, image_path, n_results, embedding_weights, output='json'):
        # TODO: refactor
        # TODO: create seperate query class, which has attributes like distances by layer, etc. This will be cleaner and allow sliders without re-running query
        query_embeddings = self._embed_query(image_path)
        query_gram_dict = self._build_query_gram_dict(query_embeddings)

        start = dt.datetime.now()
        proximal_indices = set()
        for layer_name, gram in query_gram_dict.items():
            _, closest_indices = self.index_dict[layer_name].search(gram, n_results)
            proximal_indices.update(closest_indices[0].tolist())

        dist_dict = {}
        for layer_name, gram in query_gram_dict.items():
            labels_iter_range = list(range(1, len(proximal_indices) + 1))
            labels = np.array([list(proximal_indices), labels_iter_range])
            distances = np.empty((1, len(proximal_indices)), dtype='float32')
            self.index_dict[layer_name].compute_distance_subset(
                1, faiss.swig_ptr(gram), len(proximal_indices),
                faiss.swig_ptr(distances), faiss.swig_ptr(labels))
            distances = distances.flatten()
            norm_distances = distances / max(distances)
            dist_dict[layer_name] = {idx: norm_distances[i] for i, idx in
                                     enumerate(proximal_indices)}

        print(dist_dict)

        weighted_dist_dict = {}
        for idx in proximal_indices:
            weighted_dist = sum(
                [embedding_weights[layer_name] * dist_dict[layer_name][idx] for layer_name in
                 embedding_weights])

            weighted_dist_dict[idx] = weighted_dist

        print(weighted_dist_dict)

        indices = sorted(weighted_dist_dict, key=weighted_dist_dict.get)
        results_indices = indices[:n_results]

        end = dt.datetime.now()
        index_time = (end - start).microseconds / 1000
        print(f'query time: {index_time} ms')
        print(results_indices)
        results_files = {i: self.file_mapping[i] for i in results_indices}
        results = {
            'query_img': image_path,
            'results_files': results_files,
            'similarity_weights': embedding_weights,
            'model': self.model.name,
            'lib_name': self.lib_name,
            'n_images': len(self.file_mapping),
            'invalid_paths': self.invalid_paths,
        }
        timestamp = str(dt.datetime.now())
        output_file = f'../output/results-{timestamp}'
        with open(output_file, 'w') as f:
            json.dump(results, f)

    @staticmethod
    def gram_matrix(x):
        if np.ndim(x) == 4 and x.shape[0] == 1:
            x = x[0, :]
        elif np.ndim != 3:
            # TODO: make my own error
            raise ValueError(f'')
        x = x.reshape(x.shape[-1], -1)
        gram = np.dot(x, np.transpose(x))
        return gram

    def _build_image_embedder(self, layer_range):
        conv_layers = self.model.layers[layer_range[0]: layer_range[1]]
        self.layer_names = [layer.name for layer in conv_layers]
        embedding_layers = [layer.output for layer in conv_layers]
        self.embedder = K.function([self.model.input], embedding_layers)

    def _embed_library(self, image_paths):
        inputs = []
        valid_paths = []
        self.invalid_paths = []
        for path in image_paths:
            try:
                _, x = load_image(path, self.model.input_shape[1:3])
                inputs.append(x)
                valid_paths.append(path)
            except Exception as e:
                self.invalid_paths.append(path)

        self.file_mapping = {i: f for i, f in enumerate(valid_paths)}
        lib_embeddings = [self.embedder([x, 1]) for x in inputs]
        return lib_embeddings

    def _embed_query(self, image_path):
        _, x = load_image(image_path, self.model.input_shape[1:3])
        query_embeddings = self.embedder([x, 1])
        return query_embeddings

    # TODO: split into gen_gram_matrices and _build_index, then combine gen_gram_matrices with build_query_gram_dict
    def _build_index(self, images_embeddings, buffer_size=1000):
        start = dt.datetime.now()
        self.index_dict = {}
        self.gram_list_buffer = [[] for _ in range(len(self.layer_names))]

        for i, img_embeddings in enumerate(images_embeddings):

            for k, emb in enumerate(img_embeddings):
                gram = self.gram_matrix(emb)
                gram_flat = gram.flatten()
                self.gram_list_buffer[k].append(gram_flat)

                if i == 0:
                    d = len(gram_flat)
                    self.index_dict[f'{self.layer_names[k]}'] = faiss.IndexFlatL2(d)

            if i % buffer_size == 0 and i > 0:
                self._index_buffer()

        if self.gram_list_buffer:
            self._index_buffer()
        end = dt.datetime.now()
        index_time = (end - start).microseconds / 1000
        print(f'index time: {index_time} ms')

    def _index_buffer(self):
        """
        Helper method to move data from buffer to index when `buffer_size` is
        reached
        """
        for j, gram_list in enumerate(self.gram_list_buffer):
            gram_block = np.stack(gram_list)
            self.index_dict[self.layer_names[j]].add(gram_block)
            self.gram_list_buffer = [[] for _ in range(len(self.gram_list_buffer))]

    def _build_query_gram_dict(self, img_embeddings):
        gram_dict = {}
        for i, emb in enumerate(img_embeddings):
            gram = self.gram_matrix(emb)
            gram_flat = gram.flatten()
            gram_exp = np.expand_dims(gram_flat, axis=0)
            gram_dict[self.layer_names[i]] = gram_exp
        return gram_dict
