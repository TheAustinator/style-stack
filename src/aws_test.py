from itertools import product
from keras.applications.vgg16 import VGG16

from gram_stack import GramStack


def main():
    # build parameters
    image_dir = '../data/test'
    model = VGG16(weights='imagenet', include_top=True)
    layer_range = (1, -4)
    lib_name = 'test'
    n_results = 10

    # query parameters
    weights_block1 = {
        'block1_conv1': 1,
        'block1_conv2': 1,
    }
    weights_block2 = {
        'block2_conv1': 1,
        'block2_conv2': 1,
    }
    weights_block3 = {
        'block3_conv1': 1,
        'block3_conv2': 1,
        'block3_conv3': 1,
    }
    weights_block4 = {
        'block4_conv1': 1,
        'block4_conv2': 1,
        'block4_conv3': 1,
    }
    weights_block5 = {
        'block5_conv1': 1,
        'block5_conv2': 1,
        'block5_conv3': 1,
    }
    weights_pool = {
        'block1_pool': 1,
        'block2_pool': 1,
        'block3_pool': 1,
        'block4_pool': 1,
        'block5_pool': 1,
    }
    weights_all = {
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
        'block5_conv1': 1,
        'block5_conv2': 1,
        'block5_conv3': 1,
        'block1_pool': 1,
        'block2_pool': 1,
        'block3_pool': 1,
        'block4_pool': 1,
        'block5_pool': 1,
    }
    weights_list = [weights_block1, weights_block2, weights_block3,
                    weights_block4, weights_block5, weights_pool, weights_all]
    query_images = [
        '../data/raw/125427-Dj-DX-Station-iPad-app__shot_1299500954.jpg',
        '../data/raw/735971-close-pins__cpin.png',
        '../data/raw/729487-Hazy-LA__latimes.png',
        '../data/raw/1149574-racing-app-sketch__sketch.png',
        '../data/raw/1332988-CSS-User-Agent-Selectors__blog.png',
        '../data/raw/2265403-Flat-Style-Vector-Planet-Illustration__space_with_comets-01_teaser.png',
        '../data/raw/3947574-Holidale-2017__jalapenito-dribbble-shot_teaser.png',
        '../data/raw/4507802-Mmmm__mmmm.png',
        '../data/raw/4612620-Weekend-Painting__dude.png',
        '../data/raw/4621040-Just-drawing-birdies__birdie.gif',
        '../data/raw/4666925-ICO-Landing__desktop_hd.jpg',
        ]
    search_space = list(product(query_images, weights_list))

    # build
    stack = GramStack.build(image_dir, model, layer_range)

    # save to disk
    stack.save(lib_name)

    for params in search_space:
        stack.query(params[0], n_results, params[1], write_output=True)


if __name__ == '__main__':
    main()
