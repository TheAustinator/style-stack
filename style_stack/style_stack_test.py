from itertools import product
from keras.applications.vgg16 import VGG16

from style_stack import StyleStack
from utils import plot_results

# TODO: add hypothesis and pytest testing
# TODO: package and move to tests dir


def main():
    # build parameters
    image_dir = '../data/test'
    model = VGG16(weights='imagenet', include_top=False)
    lib_name = 'raw'
    n_results = 10

    # build, save, re-load, query, and plot
    stack = StyleStack.build(image_dir, model)
    stack.save(lib_name)
    del stack
    stack = StyleStack.load(lib_name)
    query_image = '../data/raw/735971-close-pins__cpin.png'
    results = stack.query(query_image, None, n_results, write_output=True)
    plot_results(results)


if __name__ == '__main__':
    main()
