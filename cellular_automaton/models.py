""" Models of Cellular Automata. """

from typing import Sequence
from collections import Counter

import numpy as np

from . import CellularAutomaton, EdgeRule, MooreNeighbourhood


class CAImageFilter(CellularAutomaton):
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

        # print(new_cell_state, max_neighbour, min_neighbour)
        # input()

        if new_cell_state[0] < max_neighbour and new_cell_state[0] > min_neighbour:
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
            if abs(new_cell_state[0] - m) < 15:
                return new_cell_state
            else:
                return [m]

    def __del__(self):
        coordinates = self._current_state.keys()
        for coordinate, cell_c, cell_n in zip(coordinates, self._current_state.values(), self._next_state.values()):
            cell_c.neighbours = (None, )
            cell_n.neighbours = (None, )
