""" Defines cell neighbourhood. """

import enum
import operator
import itertools
import math


class EdgeRule(enum.Enum):
    IGNORE_EDGE_CELLS = 0
    IGNORE_MISSING_NEIGHBOURS_OF_EDGE_CELLS = 1
    FIRST_AND_LAST_CELL_OF_DIMENSION_ARE_NEIGHBORS = 2


class Neighbourhood:
    """ Get neighboring cells of a cell in a given position. """

    def __init__(self, edge_rule=EdgeRule.IGNORE_EDGE_CELLS, radius=1):
        """
        edge_rule: Handles behavior on cells on the edge.
        """

        self._rel_neighbours = None
        self._grid_dimensions = []
        self._radius = radius
        self.__edge_rule = edge_rule

    def calculate_cell_neighbour_coordinates(self, cell_coordinate, grid_dimensions):
        """
        Get a list of absolute coordinates for the cell neighbours.
        The EdgeRule can reduce the returned neighbour count, as in, by ignoring edge cells etc.
        :param cell_coordinate: The coordinate of the cell. In: tuple = (x, y)
        :param grid_dimensions: The dimensions of the grid, to apply the edge rule. In: array = [height, width]
        :return: List of absolute coordinates for the cells neighbours.
        """

        self.__lazy_initialize_relative_neighbourhood(grid_dimensions)
        return tuple(self._neighbours_generator(cell_coordinate))

    def __lazy_initialize_relative_neighbourhood(self, grid_dimensions):
        """ Lazy init of relative neighbourhood of a given cell. """

        self._grid_dimensions = grid_dimensions
        if self._rel_neighbours is None:
            self._create_relative_neighbourhood()

    def _create_relative_neighbourhood(self):
        self._rel_neighbours = tuple(self._neighbourhood_generator())

    def _neighbourhood_generator(self):
        for coordinate in itertools.product(range(-self._radius, self._radius + 1),
                                            repeat=len(self._grid_dimensions)):
            if self._neighbour_rule(coordinate) and coordinate != (0, ) * len(self._grid_dimensions):
                yield tuple(reversed(coordinate))

    def _neighbour_rule(self, rel_neighbor):  # pylint: disable=no-self-use, unused-argument
        return True

    def get_neighbour_by_relative_coordinate(self, neighbours, rel_coordinate):
        return neighbours[self._rel_neighbours.index(rel_coordinate)]

    def _neighbours_generator(self, cell_coordinate):
        on_edge = self.__is_coordinate_on_an_edge(cell_coordinate)
        if self.__edge_rule != EdgeRule.IGNORE_EDGE_CELLS or not on_edge:  # pylint: disable=too-many-nested-blocks
            for rel_n in self._rel_neighbours:
                if on_edge:
                    n, n_folded = zip(*[(ni + ci, (ni + di + ci) % di) for ci, ni,
                                        di in zip(cell_coordinate, rel_n, self._grid_dimensions)])
                    if self.__edge_rule == EdgeRule.FIRST_AND_LAST_CELL_OF_DIMENSION_ARE_NEIGHBORS or n == n_folded:
                        yield n_folded
                else:
                    yield tuple(map(operator.add, rel_n, cell_coordinate))

    def __is_coordinate_on_an_edge(self, coordinate):
        """ Checks to see if it is on edge by comparing the dimensions to radius and coordinates. """
        return any(not(self._radius - 1 < ci < di - self._radius)
                   for ci, di in zip(coordinate, self._grid_dimensions))


class MooreNeighbourhood(Neighbourhood):
    """
    Square neighbourhood.
    Ex.: radius 1:
        X X X X X
        X N N N X
        X N C N X
        X N N N X
        X X X X X
    """
