from typing import Sequence

import abc
import itertools
import recordclass

from cellular_automaton import Neighbourhood

CELL = recordclass.make_dataclass("Cell", ("state", "is_active", "is_dirty", "neighbours"),
                                  defaults=((0, ), True, True, (None,)))


class CellularAutomatonCreator(abc.ABC):

    def __init__(self, dimension, image, ruleset, neighbourhood: Neighbourhood, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._dimension = dimension
        self._neighbourhood = neighbourhood
        self._image = image
        self._ruleset = ruleset

        self._current_state = {}
        self._next_state = {}
        self.__make_cellular_automaton_state()

    def get_ruleset(self):
        return self._ruleset

    ruleset = property(get_ruleset)

    def get_image(self):
        return self._image

    image = property(get_image)

    def get_dimension(self):
        return self._dimension

    dimension = property(get_dimension)

    def __make_cellular_automaton_state(self):
        self.__make_cells()
        self.__add_neighbours()

    def __make_cells(self):
        for coordinate in itertools.product(*[range(d) for d in self._dimension]):
            cell_state = self.init_cell_state(coordinate)
            self._current_state[coordinate] = CELL(cell_state)
            self._next_state[coordinate] = CELL(cell_state)

    def __add_neighbours(self):
        """ Sets neighbours for each cell in the state dictionary.
        Loops through 'zipped' coordinates and dictionary values of both next and current state.
        :param coordinate: (x, y)
        :param cell_c: CELL of current state
        :param cell_n: CELL of next state 
        """
        calculate_cell_neighbour_coordinates = self._neighbourhood.calculate_cell_neighbour_coordinates
        coordinates = self._current_state.keys()
        for coordinate, cell_c, cell_n in zip(coordinates, self._current_state.values(), self._next_state.values()):
            n_coord = calculate_cell_neighbour_coordinates(
                coordinate, self._dimension)
            cell_c.neighbours = list([self._current_state[nc]
                                      for nc in n_coord])
            cell_n.neighbours = list([self._next_state[nc] for nc in n_coord])

    def init_cell_state(self, cell_coordinate: Sequence) -> Sequence:  # pragma: no cover
        """ Will be called to initialize a cells state.
        :param cell_coordinate: Cells coordinate.
        :return: Iterable that represents the initial cell state
        """
        raise NotImplementedError


class CellularAutomaton(CellularAutomatonCreator, abc.ABC):

    def __init__(self, neighbourhood: Neighbourhood, *args, **kwargs):
        super().__init__(neighbourhood=neighbourhood, *args, **kwargs)
        self._evolution_step = 0
        self._active = True

    def is_active(self):
        return self._active

    def reactive(self):

        for cell in self._current_state.values():
            cell.is_active = True
            cell.is_dirty = True
        self._active = True

    active = property(is_active)

    def get_cells(self):
        return self._current_state

    def set_cells(self, cells):
        for (coordinate, c_cell), n_cell in zip(self._current_state.items(), self._next_state.values()):
            new_cell_state = cells[coordinate].state
            c_cell.state = new_cell_state
            n_cell.state = new_cell_state

    cells = property(get_cells, set_cells)

    def get_evolution_step(self):
        return self._evolution_step

    evolution_step = property(get_evolution_step)

    def evolve(self, times=1):
        """
        Evolve all cells a number of times.
        :param times: Number of evolution steps processed with one call of this method.
        """

        for _ in itertools.repeat(None, times):
            self._active = False
            self.__evolve_cells(self._current_state, self._next_state)
            self._current_state, self._next_state = self._next_state, self._current_state
            self._evolution_step += 1

    def __evolve_cells(self, this_state, next_state):
        evolve_cell = self.__evolve_cell
        evolution_rule = self.evolve_rule
        for old, new in zip(this_state.values(), next_state.values()):
            if old.is_active:
                new_state = evolution_rule(
                    old.state.copy(), [n.state for n in old.neighbours])
                old.is_active = False
                evolve_cell(old, new, new_state)

    def __evolve_cell(self, old, cell, new_state):
        changed = new_state != old.state
        cell.state = new_state
        cell.is_dirty |= changed
        old.is_dirty |= changed
        self._active |= changed
        if changed:
            cell.is_active = True
            for n in cell.neighbours:
                n.is_active = True

    def evolve_rule(self, last_cell_state: Sequence, neighbours_last_states: Sequence) -> Sequence:  # pragma: no cover
        """ Calculates and sets new state of 'cell'.
        A cells evolution will only be called if it or at least one of its neighbours has changed last evolution_step.
        :param last_cell_state:         The cells state previous to the evolution step.
        :param neighbours_last_states:   The cells neighbours current states.
        :return: New state.             The state after this evolution step
        """
        raise NotImplementedError
