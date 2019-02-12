import datetime as dt
import faiss
import glob
import json
import keras.applications as apps
import keras.backend as K
from math import ceil
import numpy as np
import os
import random
import re

from utils import load_image, get_image_paths


class StyleStack:
    """
    This class is used to do style similarity search of a query image against
    a libaray of images. The search uses the l2 difference between the gram
    matrices of user-selected embeddings from a user specified convolutional
    network. The similarity search uses Facebook AI Research's faiss library
    to for speed and scalability, and the `StyleStack` class acts as a high level
    ORM to the index.

    This class is not instantiated directly, but constructed via either the
    `StyleStack.build` class method to build a new set of indexes from raw
    images, or the `StyleStack.load` constructor to load a set of indexes from
    disk. More images can always be added to an existing index using the `add`
    method on an instance of `StyleStack`. The `query` method is used to get the
    filepaths of the most similar images. The weighting of different layer
    embeddings in the similarity search can be adjusted at query time.

    Example:
        Building a StyleStack
        >>> image_dir = '../data/my_data'
        >>> model = apps.vgg16.VGG16(weights='imagenet', include_top=False)
        >>> layer_range = ('block1_conv1', 'block2_pool')
        >>> stack = StyleStack.build(image_dir, model, layer_range)
        Saving
        >>> lib_name = 'library_name'
        >>> stack.save(lib_name)
        Loading
        >>> stack = StyleStack.load(lib_name)
        Querying
        >>> # any embeddings not in embedding_weights will not be used
        >>> image_path = '../data/my_data/cat_painting.jpg'
        >>> embedding_weights = {
        >>>     'block1_conv1': 1,
        >>>     'block3_conv2': 0.5,
        >>>     'block3_pool': .25
        >>> }
        >>> n_results = 5
        >>> results = stack.query(image_path, n_results, embedding_weights,
        >>>                       write_output=True)
    """

    models = {
        'densenet121': apps.densenet.DenseNet121,
        'densenet169': apps.densenet.DenseNet169,
        'densenet201': apps.densenet.DenseNet201,
        'inceptionv3': apps.inception_v3.InceptionV3,
        'inceptionresnetv2': apps.inception_resnet_v2.InceptionResNetV2,
        'mobilenet': apps.mobilenet.MobileNet,
        'mobilenetv2': apps.mobilenet_v2.MobileNetV2,
        'nasnetlarge': apps.nasnet.NASNetLarge,
        'nasnetmobile': apps.nasnet.NASNetMobile,
        'resnet50': apps.resnet50.ResNet50,
        'vgg16': apps.vgg16.VGG16,
        'vgg19': apps.vgg19.VGG19,
        'xception': apps.xception.Xception,
    }

    def __init__(self):
        self.valid_paths = []
        self.invalid_paths = []
        self.lib_name = None
        self.vector_buffer_size = None
        self.index_buffer_size = None
        self._file_mapping = None
        self._partitioned = None

    @classmethod
    def build(cls, image_dir, model, layer_range, vector_buffer_size=100,
              index_buffer_size=6500):
        """
        Use this constructor when you do not have a preexisting gram matrix
        library built by another instance of `StyleStack`.

        This is the first of two constructors for `StyleStack`, which builds a
        set of faiss indexes, one for each embedding layer of the model. Each
        index is specific to the model and layer that were used to embed it, so
        a new `StyleStack` should be built if the `model` is changed or if
        any of the layers used to `query` are not in `layer_range` of the prior
        gram matrix library. However, if the layers used to query are a subset
        of `layer_range` and the `model` and images are the same, use `load`.

        Args:
            image_dir (str): directory containing images to be indexed

            model (keras.engine.training.Model): model from which to extract
                embeddings.

            layer_range (None, iterable[str]): A two item iterable containing the
                desired first and last layer names to be inclusively used as
                layer selection bounds. Alternatively, `None` uses all layers
                except for input. Note that the gram matrix computations do not
                work on dense layers, so if the model has dense layers, they
                should be excluded with this argument or loaded with
                `include_top=False` in keras.

            vector_buffer_size (int): number of embeddings to load into memory
                before indexing them. Reduce this if your system runs out of
                memory before printouts that some items have been indexed.

            index_buffer_size (int): number of files to index before forcing indexes
                to be saved to disk. If the number of image files is less than
                this, than the indexes will be held in memory. Otherwise, the
                partitioned indexes are automatically saved to disk when they
                reach the size of `index_buffer_size`. Reduce this when running
                out of memory subsequent to printouts that some items have been
                indexed.

        Returns:
            inst (cls): instance of `cls` built from the gram matrices of the
                embeddings generated by the layers of the `model` specified by
                the `layer_range`. Note that the underlying indexes are not
                saved unless the `inst.save` method is run.
        """
        inst = cls()
        inst.lib_name = None
        inst.vector_buffer_size = vector_buffer_size
        inst.index_buffer_size = index_buffer_size
        image_paths = get_image_paths(image_dir)
        inst.model = model
        inst._build_image_embedder(layer_range)
        inst._embedding_gen = inst._gen_lib_embeddings(image_paths)
        inst._build_index(inst._embedding_gen)
        return inst

    @classmethod
    def load(cls, lib_name, layer_range, model=None):
        """

        Args:
            lib_name:

            layer_range:

            model: Model to embed query images. It must be the same as the model
                used to embed the reference library. If `None`, the model name
                will be gathered from the metadata, and it will be loaded from
                `keras.applications` with `weights='imagenet'` and
                `include_top=False`. If the model used to build the original
                `StyleStack` is not part of `keras.applications` or did not use
                imagenet weights, the model will not be generated correctly from
                metadata and must be passed in via this argument

        Returns:

        """
        input_dir = f'../data/indexes/{lib_name}/'
        inst = cls()
        cls.lib_name = lib_name

        # invalid paths have already been filtered out
        inst.invalid_paths = None

        # load metadata
        with open(os.path.join(input_dir, 'meta.json')) as f:
            json_str = json.load(f)
            metadata = {str(k): v for k, v in json_str.items()}
        if model is None:
            model_str = metadata['model']
            model_cls = StyleStack.models[model_str]
            model = model_cls(weights='imagenet', include_top=False)

        # load file mapping
        with open(os.path.join(input_dir, 'file_mapping.json')) as f:
            json_str = json.load(f)
            inst.file_mapping = {int(k): str(v) for k, v in json_str.items()}

        # load gram matrix indexes
        index_paths = glob.glob(os.path.join(input_dir, 'grams-*.index'))

        # check for partitioning
        if 'part' in index_paths[0]:
            inst.partitioned = True

        # TODO: finish this
        if inst.partitioned:
            partition_dict = {

            }

        inst.index_dict = {}
        for f in index_paths:
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

    def query(self, image_path, embedding_weights=None, n_results=5,
              write_output=True):
        # TODO: refactor
        # TODO: create seperate query class, which has attributes like distances by layer, etc. This will be cleaner and allow sliders without re-running query
        query_embeddings = self._embed_image(image_path)
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
        results_files = [self.file_mapping[i] for i in results_indices]
        results = {
            'query_img': image_path,
            'results_files': results_files,
            'similarity_weights': embedding_weights,
            'model': self.model.name,
            'lib_name': self.lib_name,
            'n_images': len(self.file_mapping),
            'invalid_paths': self.invalid_paths,
        }
        if write_output:
            timestamp = str(dt.datetime.now())
            output_dir = f'../output/'
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            output_file = os.path.join(output_dir, f'results-{timestamp}')
            with open(output_file, 'w') as f:
                json.dump(results, f)
        return results

    @property
    def metadata(self):
        return {
            'model': self.model.name,
            'layer_names': self.layer_names,
            'paritioned': self.partitioned,
        }

    @property
    def partitioned(self):
        if self._partitioned is None:
            input_dir = f'../data/indexes/{self.lib_name}/'
            indexes = glob.glob(os.path.join(input_dir, 'grams-*.index'))
            if 'part' in indexes[0]:
                return True
            else:
                return False
        else:
            return self._partitioned

    @partitioned.setter
    def partitioned(self):


    @property
    def file_mapping(self):
        if self._file_mapping:
            pass
        else:
            self._file_mapping = {i: f for i, f in enumerate(self.valid_paths)}
        return self._file_mapping

    @file_mapping.setter
    def file_mapping(self, value):
        self._file_mapping = value

    @staticmethod
    def gram_vector(x):
        if np.ndim(x) == 4 and x.shape[0] == 1:
            x = x[0, :]
        elif np.ndim != 3:
            # TODO: make my own error
            raise ValueError(f'')
        x = x.reshape(x.shape[-1], -1)
        gram_mat = np.dot(x, np.transpose(x))
        mask = np.triu_indices(len(gram_mat), 1)
        gram_mat[mask] = None
        gram_vec = gram_mat.flatten()
        gram_vec = gram_vec[~np.isnan(gram_vec)]
        return gram_vec

    def _build_query_gram_dict(self, img_embeddings):
        gram_dict = {}
        for i, emb in enumerate(img_embeddings):
            gram_vec = self.gram_vector(emb)
            gram_vec = np.expand_dims(gram_vec, axis=0)
            gram_dict[self.layer_names[i]] = gram_vec
        return gram_dict

    def _build_image_embedder(self, layer_range=None):
        layer_names = [layer.name for layer in self.model.layers]
        if layer_range:
            slice_start = layer_names.index([layer_range[0]])
            slice_end = layer_names.index([layer_range[1]]) + 1
            chosen_layer_names = layer_names[slice_start:slice_end]
            chosen_layers = [layer for layer in self.model.layers
                             if layer.name in chosen_layer_names]
        else:
            chosen_layer_names = layer_names[1:]
            chosen_layers = self.model.layers[1:]
        self.layer_names = chosen_layer_names
        embedding_layers = [layer.output for layer in chosen_layers]
        self.embedder = K.function([self.model.input], embedding_layers)

    # TODO: output successful embeddings to make resumable
    def _gen_lib_embeddings(self, image_paths):
        for path in image_paths:
            try:
                image_embeddings = self._embed_image(path)
                self.valid_paths.append(path)
                yield image_embeddings

            except Exception as e:
                # TODO: add logging
                print(f'Embedding error: {e.args}')
                self.invalid_paths.append(path)
                continue

    def _embed_image(self, image_path):
        if self.model.input_shape[1]:
            _, x = load_image(image_path, self.model.input_shape[1:3])
        else:
            _, x = load_image(image_path, target_size=(224, 224))

        image_embeddings = self.embedder([x, 1])
        return image_embeddings

    # TODO: split into gen_gram_matrices and _build_index, then combine gen_gram_matrices with build_query_gram_dict
    def _build_index(self, img_embedding_gen):
        start = dt.datetime.now()
        in_memory = True
        part_num = 0
        self.index_dict = {}
        self.vector_buffer = [[] for _ in range(len(self.layer_names))]
        for i, img_embeddings in enumerate(img_embedding_gen):

            for k, emb in enumerate(img_embeddings):
                gram_vec = self.gram_vector(emb)
                self.vector_buffer[k].append(gram_vec)

                if i == 0:
                    d = len(gram_vec)
                    self.index_dict[f'{self.layer_names[k]}'] = \
                        faiss.IndexFlatL2(d)

            if i % self.vector_buffer_size == 0 and i > 0:
                self._index_vectors()
                print(f'images {i - self.vector_buffer_size} - {i} indexed')

            if i % self.index_buffer_size == 0 and i > 0:
                in_memory = False
                part_num = ceil(i / self.index_buffer_size)
                self._save_indexes(self.lib_name, part_num)

        if self.vector_buffer:
            self._index_vectors()
            if not in_memory:
                part_num += 1
                self._save_indexes(self.lib_name, part_num)

        end = dt.datetime.now()
        index_time = (end - start).microseconds / 1000
        print(f'index time: {index_time} ms')

    def _index_vectors(self):
        """
        Helper method to move data from buffer to index when
        `vector_buffer_size` is reached
        """
        for j, gram_list in enumerate(self.vector_buffer):
            gram_block = np.stack(gram_list)
            self.index_dict[self.layer_names[j]].add(gram_block)
            self.vector_buffer = [[] for _ in range(len(self.vector_buffer))]

    def _save_indexes(self, lib_name, part_num):
        if self.vector_buffer:
            self._index_vectors()

        self.lib_name = lib_name
        output_dir = f'../data/indexes/{lib_name}/'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        for layer_name, index in self.index_dict.items():
            filename = f'grams-{layer_name}-part_{part_num}.index'
            filepath = os.path.join(output_dir, filename)
            faiss.write_index(index, filepath)
            self.index_dict = {}

        # save metadata
        if part_num == 1:
            metadata = {
                'model': self.model.name,
                'layer_names': self.layer_names,
            }
            metadata_path = os.path.join(output_dir, 'meta.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f)
