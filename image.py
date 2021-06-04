from image_processing.io import read_preprocess
import os

from typing import Sequence
from collections import Counter

import cv2 as cv
import numpy as np
from scipy.ndimage.measurements import mean
from skimage.metrics import structural_similarity as ssim

from matplotlib import pyplot as plt

from cellular_automaton import CellularAutomaton, EdgeRule, MooreNeighbourhood
from image_processing import save_img


class ImageCA(CellularAutomaton):
    """
    Implementation of cellular automaton using grayscale image pixel values as initial state.
    """

    def __init__(self, dimension, image, ruleset):
        super().__init__(dimension=dimension, image=image, ruleset=ruleset,
                         neighbourhood=MooreNeighbourhood(EdgeRule.IGNORE_EDGE_CELLS))

    def init_cell_state(self, cell_coordinate: Sequence) -> Sequence:  # pragma: no cover
        x, y = cell_coordinate
        init = self._image[x][y]
        return [init]

    def evolve_rule(self, last_cell_state, neighbours_last_states):
        """
        Change cell state if neighbours match a rule in ruleset.
        New state will be the average of neighbour cells.
        """

        new_cell_state = last_cell_state

        neighbours = [n[0] for n in neighbours_last_states]

        if neighbours == []:
            return new_cell_state

        max_neighbour = max(neighbours)
        min_neighbour = min(neighbours)

        states_neighbours = Counter(neighbours)
        num_states_neighbours = len(states_neighbours)

        if new_cell_state <= max_neighbour and new_cell_state > min_neighbour:
            return new_cell_state
        elif max_neighbour == min_neighbour or num_states_neighbours <= 2:
            if min_neighbour != 0:
                return [min_neighbour]
            elif max_neighbour != 255:
                return [max_neighbour]
            else:
                return new_cell_state
        else:
            for _ in range(states_neighbours[max_neighbour]):
                neighbours.remove(max_neighbour)
            for _ in range(states_neighbours[min_neighbour]):
                neighbours.remove(min_neighbour)
            m = np.mean(neighbours)
            if abs(new_cell_state - m) < 15:
                return new_cell_state
            else:
                return [m]

    def __del__(self):
        coordinates = self._current_state.keys()
        for coordinate, cell_c, cell_n in zip(coordinates, self._current_state.values(), self._next_state.values()):
            cell_c.neighbours = (None, )
            cell_n.neighbours = (None, )


def configure_and_save(grid, rs, filename, t):
    h = grid.shape[0]
    w = grid.shape[1]

    ca_dimension = [h, w]
    ca_save = ImageCA(ca_dimension, grid, rs)
    ca_save.evolve(times=t)

    keys = list(ca_save.cells.keys())
    cells = list(ca_save.cells.values())

    image = []
    for row in range(0, len(keys), w):
        image_row = np.array([cell.state[0] for cell in cells[row:row + w]])
        image.append(image_row)

    save_img("./results/", filename, np.asarray(image))


def compare_ssim(im_compare, im_predict):
    im_ssim = ssim(im_compare, im_predict, data_range=im_predict.max(
    ) - im_predict.min(), multichannel=True)

    return im_ssim


def view_decomposed():
    img = cv.imread("./samples/processed/satellite_4.jpg", cv.IMREAD_GRAYSCALE)
    img = read_preprocess("./samples/processed/satellite_4.jpg", resize=True, height_resize=500)
    imgs = dict()
    imgs["Imagem Original"] = img
    for x in range(51, 255, 51):
        ret, thresh = cv.threshold(img, x, 255, cv.THRESH_BINARY)
        imgs[f'Limiar {x}'] = thresh
    
    for i, (title, img) in enumerate(imgs.items()):
        plt.subplot(2,3,i+1), plt.imshow(img, 'gray', vmin=0, vmax=255)
        plt.title(title)
        plt.xticks([]),plt.yticks([])
    plt.show()


# noisy = cv.imread(os.path.abspath("./") +
#                   "/samples/noisy/gaussian/satellite_2.jpg", cv.IMREAD_GRAYSCALE)

# configure_and_save(noisy.copy(), {}, 'none.jpg', 100)

view_decomposed()
