from keras.applications.vgg16 import VGG16

from gram_stack import StyleStack


def main():
    build_index_example()
    load_and_query_example()


def build_index_example():
    # parameters
    image_dir = '../data/test'
    model = VGG16(weights='imagenet', include_top=True)
    layer_range = (1, -4)
    lib_name = 'test_1'

    # build
    stack = StyleStack.build(image_dir, model, layer_range)

    # save to disk
    stack.save(lib_name)


def load_and_query_example():
    # query parameters
    lib_name = 'test_1'
    layer_range = (1, -4)
    n_results = 10
    query_path = '../data/test/142362-Louis-Vuitton-Signature-Collection__pean.jpg'
    embedding_weights = {
        'block1_conv1': 1,
        'block1_conv2': 1,
    }

    # load from disk
    stack = StyleStack.load(lib_name, layer_range)

    # query
    results = stack.query(query_path, embedding_weights, n_results,
                          write_output=False)
    print(results)


if __name__ == '__main__':
    main()
