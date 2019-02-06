import faiss
from keras.applications.vgg16 import VGG16
import numpy as np

from utils import get_image_paths
from gram_matrices import gen_images_embeddings, gram_matrix, plot_results


def main():
    model = VGG16(weights='imagenet', include_top=True)
    image_paths = get_image_paths('../data/test')
    images_embeddings, layer_names, file_mapping = gen_images_embeddings(image_paths, model)

    lib_name = 'test_1'

    # create library
    index_dict = {}
    gram_list_list = [[] for _ in range(len(layer_names))]
    for i, img_embeddings in enumerate(images_embeddings):

        for k, emb in enumerate(img_embeddings):
            gram = gram_matrix(emb)
            gram_flat = gram.flatten()
            gram_list_list[k].append(gram_flat)

            if i == 0:
                d = len(gram_flat)
                index_dict[f'{layer_names[k]}'] = faiss.IndexFlatL2(d)

        if i % 100 == 0 and i > 0:
            for j, gram_list in enumerate(gram_list_list):
                gram_stack = np.stack(gram_list)
                index_dict[layer_names[j]].add(gram_stack)
                gram_list_list = [[] for _ in range(len(gram_list_list))]

    if gram_list_list:
        for j, gram_list in enumerate(gram_list_list):
            gram_stack = np.stack(gram_list)
            index_dict[layer_names[j]].add(gram_stack)
            gram_list_list = [[] for _ in range(len(gram_list_list))]

    # -------
    query_idx = 0
    k = 4

    query_gram_dict = {layer_name: index[query_idx] for layer_name, index in
                       index_dict.items()}

    # query
    for layer_name, gram in query_gram_dict:
        D, I = index.search(gram, k)
    # ------
    # load_gram_lib
    start = dt.datetime.now()
    file_index, gram_lib = load_gram_lib(lib_name)
    for k, gram_stack in gram_lib.items():
        print(gram_stack.shape)



if __name__ == '__main__':
    main()
