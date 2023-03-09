#!/usr/bin/python

import numpy as np
from tronproblem import *
from trontypes import CellType, PowerupType
import random, math
import collections
from queue import PriorityQueue

# Throughout this file, ASP means adversarial search problem.

class StudentBot:
    def __init__(self):
        self.dir = ["U", "D", "L", "R"]
        self.BOT_NAME = "bruh moment"
    
    def abc_max(self, asp, state, to_move, alpha, beta, depth, max_depth):
        board = state.board
        loc = state.player_locs[0]
        if asp.is_terminal_state(state):
            return (asp.evaluate_state(state)[to_move], None)
        safe_moves = list(TronProblem.get_safe_actions(board, loc))
        v, move = float("-inf"), None
        if depth == max_depth:
            return (self.voronoi_heuristic(state, to_move), None)
        for act in self.dir:
            if act not in safe_moves:
                continue
            # nxt_loc = TronProblem.move(loc, act)
            nxt_state = asp.transition(state, act)
            v2, a2 = self.abc_min(asp, nxt_state, to_move, alpha, beta, depth + 1, max_depth)
            if v2 > v:
                v = v2
                move = act
            if v >= beta:
                return (v, move)
            alpha = max(alpha, v)
        return (v, move)
        
    def abc_min(self, asp, state, to_move, alpha, beta, depth, max_depth):
        board = state.board
        loc = state.player_locs[1]
        if asp.is_terminal_state(state):
            return (asp.evaluate_state(state)[to_move], None)
        safe_moves = list(TronProblem.get_safe_actions(board, loc))
        v, move = float("inf"), None
        if depth == max_depth:
            return (self.voronoi_heuristic(state, to_move), None)
        for act in self.dir:
            if act not in safe_moves:
                continue
            # nxt_loc = TronProblem.move(loc, act)
            nxt_state = asp.transition(state, act)
            v2, a2 = self.abc_max(asp, nxt_state, to_move, alpha, beta, depth + 1, max_depth)
            if v2 < v:
                v = v2
                move = act
            if v <= alpha:
                return (v, move)
            beta = min(beta, v)
        return (v, move)

    def get_neighbors(self, r, c, state):
        total_row, total_col = np.asarray(state.board).shape
        neighbors = []
        if r < total_row - 1:
            cell = state.board[r+1][c]
            if cell != CellType.WALL and cell != CellType.BARRIER:
                neighbors.append((r+1, c))
        if r > 0:
            cell = state.board[r-1][c]
            if cell != CellType.WALL and cell != CellType.BARRIER:
                neighbors.append((r-1, c))
        if c < total_col - 1:
            cell = state.board[r][c+1]
            if cell != CellType.WALL and cell != CellType.BARRIER:
                neighbors.append((r, c+1))
        if c > 0:
            cell = state.board[r][c-1]
            if cell != CellType.WALL and cell != CellType.BARRIER:
                neighbors.append((r, c-1))
        return neighbors

    def dijkstra(self, state, to_move, r, c):
        #r, c is player1 location
        row, col = np.asarray(state.board).shape
        distances = np.full((row, col), np.inf) #initialize distances from player to square to infinity.
        distances[r, c] = 0
        visited = np.full((row, col), False)
        neighbors = self.get_neighbors(r, c, state)
        for n in neighbors:
            r, c = n
            distances[r,c] = 1 
        Q = collections.deque(neighbors)
        while len(Q) != 0:
            current_r, current_c = Q.popleft()
            new_distance = distances[current_r, current_c] + 1
            for n in self.get_neighbors(current_r, current_c, state):
                newr, newc = n
                cell = state.board[newr][newc]
                if new_distance < distances[newr,newc]:
                    distances[newr,newc] = new_distance
                if not visited[newr, newc]:
                    visited[newr, newc] = True
                    Q.append(n)
        print(distances)
        return distances

    def dijkstra_priority(self, state, s_r, s_c):
        row, col = len(state.board), len(state.board[0])
        distances = np.full((row, col), np.inf)
        distances[s_r, s_c] = 0
        pq = PriorityQueue()
        pq.put((0, (s_r, s_c)))
        visited = set()
        while not pq.empty():
            dist, cv = pq.get()
            visited.add(cv)
            for n in self.get_neighbors(cv[0], cv[1], state):
                nxt_r, nxt_c = n
                nxt_dist = dist + 1
                if n not in visited and (nxt_dist < distances[nxt_r, nxt_c]):
                    distances[nxt_r, nxt_c] = nxt_dist
                    pq.put((nxt_dist, (nxt_r, nxt_c)))
        return distances 

    def voronoi_heuristic(self, state, to_move):
        row, col = np.asarray(state.board).shape
        p1_loc = state.player_locs[to_move]
        p2_loc = state.player_locs[state.ptm]
        p1d = self.dijkstra_priority(state, p1_loc[0], p1_loc[1])
        p2d = self.dijkstra_priority(state, p2_loc[0], p2_loc[1])
        maxcost = row + col 
        p1region = 0
        p2region = 0
        for r in range(row):
            for c in range(col):
                if p1d[r,c] < p2d[r,c] and p1d[r,c] <= maxcost:
                    p1region += 1
                if p2d[r,c] < p1d[r,c] and p2d[r,c] <= maxcost:
                    p2region += 1
        def sigmoid(x):
            return 1 / (1 + math.exp(-x))
        return (p1region - p2region) / (p1region + p2region)

    def decide(self, asp):
        state = asp.get_start_state()
        locs = state.player_locs
        board = state.board
        ptm = state.ptm
        loc = locs[ptm]
        # print((self.abc_max(asp, state, ptm, float("-inf"), float("inf"), 0, 5, loc)))
        return self.abc_max(asp, state, ptm, float("-inf"), float("inf"), 0, 5)[1]

    def cleanup(self):
        pass

class RandBot:
    """Moves in a random (safe) direction"""

    def decide(self, asp):
        """
        Input: asp, a TronProblem
        Output: A direction in {'U','D','L','R'}
        """
        state = asp.get_start_state()
        locs = state.player_locs
        board = state.board
        ptm = state.ptm
        loc = locs[ptm]
        possibilities = list(TronProblem.get_safe_actions(board, loc))
        if possibilities:
            return random.choice(possibilities)
        return "U"
    
    def cleanup(self):
        pass


class WallBot:
    """Hugs the wall"""

    def __init__(self):
        order = ["U", "D", "L", "R"]
        random.shuffle(order)
        self.order = order

    def cleanup(self):
        order = ["U", "D", "L", "R"]
        random.shuffle(order)
        self.order = order

    def decide(self, asp):
        """
        Input: asp, a TronProblem
        Output: A direction in {'U','D','L','R'}
        """
        state = asp.get_start_state()
        locs = state.player_locs
        board = state.board
        ptm = state.ptm
        loc = locs[ptm]
        possibilities = list(TronProblem.get_safe_actions(board, loc))
        if not possibilities:
            return "U"
        decision = possibilities[0]
        for move in self.order:
            if move not in possibilities:
                continue
            next_loc = TronProblem.move(loc, move)
            if len(TronProblem.get_safe_actions(board, next_loc)) < 3:
                decision = move
                break
        return decision
