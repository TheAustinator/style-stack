import datetime as dt

from utils import load_image, get_concatenated_images, get_image_paths
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import random


def main():
    paths = get_image_paths('../data/test')
    query_indices = [int(len(paths) * random.random()) for _ in range(5)]

    timestamp = dt.datetime.now()
    pdf_pages = PdfPages(f'../output/plotting_test_{timestamp}.pdf')
    for i in range(5):
        query_image_idx = 1
        closest_image_indices = range(2, 6)
        query_image = get_concatenated_images(paths, [query_image_idx])
        results_image = get_concatenated_images(paths, closest_image_indices)

        fig, (plot_1, plot_2) = plt.figure(figsize=(8.27, 11.69), dpi=100)
        plt.subplot2grid((2, 1), (0, 0))
        plt.imshow(query_image)
        plt.title(f'query image {query_image_idx}')
        plt.subplot2grid((2, 1), (1, 0))
        plt.imshow(results_image)
        plt.title(f'result images: {closest_image_indices}')
        plt.tight_layout()
        pdf_pages.savefig(fig)
    pdf_pages.close()


if __name__ == '__main__':
    main()
