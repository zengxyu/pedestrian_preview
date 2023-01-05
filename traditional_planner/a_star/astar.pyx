import copy
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict

class Cell:
    __slots__ = "x", "y", "g_cost", "h_cost", "total_cost", "parent"

    def __init__(self, x, y, parent):
        self.x = x
        self.y = y
        self.g_cost = 0
        self.h_cost = 0
        self.total_cost = 0
        self.parent = parent


class AStar:
    __slots__ = "_occupancy_map", "open_list", "close_list"

    def __init__(
            self,
            occupancy_map: np.ndarray,
    ):
        self._occupancy_map = occupancy_map
        self.open_list = {}
        self.close_list = {}

    def search_path(self, start_pos: Tuple[int, int], end_pos: Tuple[int, int]):
        """
        The main function of A* algorithm

        Returns
        -------

        """
        self.open_list = {}
        self.close_list = {}
        start_cell = Cell(start_pos[0], start_pos[1], None)
        self.open_list[start_pos] = start_cell
        while True:
            key = self.select_cell_to_explore()
            if key is None:
                return None
            cell = self.open_list[key]
            self.open_list.pop(key)
            self.close_list[key] = cell

            if cell.x == end_pos[0] and cell.y == end_pos[1]:
                return self.build_path(cell)

            self.explore_neighbors(cell, end_pos)

    def compute_cost(self, cell: Cell, end_pos):
        """
        Compute g_cost, h_cost, and total cost for the cell
        Parameters
        ----------
        cell

        Returns
        -------

        """
        # real cost from start position to (x,y)
        parent = cell.parent
        gn = parent.g_cost + np.sqrt(
            (cell.x - parent.x) ** 2 + (cell.y - parent.y) ** 2
        )
        # heuristic cost from (x,y) to end position, use euclidean distance here
        hn = np.sqrt((end_pos[0] - cell.x) ** 2 + (end_pos[1] - cell.y) ** 2)
        cell.g_cost = gn
        cell.h_cost = hn
        cell.total_cost = gn + hn

    def explore_neighbors(self, cell, end_pos):
        """
        Explore neighbors for a cell, add it into open_list or update the g_cost if it's already in open_list
        Parameters
        ----------
        cell

        Returns
        -------

        """
        for i in range(-1, 2, 1):
            for j in range(-1, 2, 1):
                # neighbor coordinate
                nx, ny = cell.x + i, cell.y + j
                # ignore current cell if it is invalid or already passed
                if (
                        (i == 0 and j == 0)
                        or not self.is_valid(nx, ny)
                        or self.is_in_cell_list(nx, ny, self.close_list)
                ):
                    continue
                else:
                    # if neighbor is already in the list, compare the old G cost and the new G cost
                    if (nx,ny) in self.open_list:
                        neighbor = self.open_list[(nx, ny)]
                        new_g_cost = cell.g_cost + np.sqrt(
                            (neighbor.x - cell.x) ** 2 + (neighbor.y - cell.y) ** 2
                        )
                        if new_g_cost < neighbor.g_cost:
                            neighbor.g_cost = new_g_cost
                            neighbor.total_cost = new_g_cost + neighbor.h_cost
                            neighbor.parent = cell
                    # neighbor is not in open_list, add it
                    else:
                        neighbor = Cell(nx, ny, cell)
                        self.compute_cost(neighbor, end_pos)
                        self.open_list[(nx, ny)] = neighbor

    def select_cell_to_explore(self) -> int:
        """
        Select a cell with minimum total cost from open_list to expand

        Returns cell index in the open_list
        -------

        """
        best_key = None
        cdef int min_total_cost = 0
        cdef int i = 0
        for key, cell in self.open_list.items():
            if i == 0:
                best_key = key
                min_total_cost = cell.total_cost
            elif cell.total_cost < min_total_cost:
                best_key = key
                min_total_cost = cell.total_cost
            i += 1
        return best_key

    def build_path(self, cell: Cell) -> List[Tuple[int, int]]:
        """
        build a final path from start to end
        Parameters
        ----------
        cell : end cell

        Returns
        -------

        """
        path = []
        while cell is not None:
            path.append((cell.x, cell.y))
            cell = cell.parent
        path.reverse()
        return path

    def plot_path(self, path: List[Tuple[int, int]]):
        if path is None:
            return
        map = copy.deepcopy(self._occupancy_map)
        map = map.astype(np.float)
        for x, y in path:
            map[x, y] = 0.8
        plt.figure()
        plt.imshow(map)
        plt.show()

    def is_valid(self, x, y) -> bool:
        """
        check if the cell is free and inside the map.
        (That is, invalid when the position is out of bound or an obstacle)

        Parameters
        ----------
        x :
        y :

        Returns
        -------
                True if the cell is free and inside the map

        """
        if x < 0 or y < 0:
            return False
        elif x >= self._occupancy_map.shape[0] or y >= self._occupancy_map.shape[1]:
            return False
        else:
            return not self._occupancy_map[x][y]

    def is_in_cell_list(self, x: int, y: int, cell_list: Dict) -> bool:
        """
        Check if a cell with given value is already in the cell_list
        Parameters
        ----------
          x : row index of the grid map
        y : col index of the grid map
        cell_list : open_list or close_list

        Returns
        -------

        """
        return (x, y) in cell_list

