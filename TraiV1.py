# Qtsumego Trajectory-Based Born Machine

"""
Trajectory-Based Quantum Go Life Analyzer
----------------------------------------

Main conceptual upgrade:
Instead of learning probabilities over SINGLE MOVES,
we learn probabilities over COMPLETE TACTICAL TRAJECTORIES.

This approximates tactical reading in Go.

Author: Alberto Vari + ChatGPT integration
"""

import numpy as np
import itertools
import time
from typing import List, Dict, Tuple
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.circuit.library import RealAmplitudes

# =========================================================
# 1. BOARD UTILITIES
# =========================================================

LETTERS = "ABCDEFGHI"


def coord_to_index(coord, board_size):
    col = LETTERS.index(coord[0].upper())
    row = int(coord[1:]) - 1
    return row * board_size + col


def index_to_coord(index, board_size):
    row, col = divmod(index, board_size)
    return f"{LETTERS[col]}{row + 1}"


def get_neighbors(position, board_size):
    row, col = divmod(position, board_size)
    neighbors = []

    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nr, nc = row + dr, col + dc
        if 0 <= nr < board_size and 0 <= nc < board_size:
            neighbors.append(nr * board_size + nc)

    return neighbors


# =========================================================
# 2. GO BOARD
# =========================================================

class GoBoard:
    EMPTY, BLACK, WHITE = 0, 1, 2

    def __init__(self, board_size=9):
        self.size = board_size
        self.board = np.zeros((board_size, board_size), dtype=int)

    def copy(self):
        newb = GoBoard(self.size)
        newb.board = self.board.copy()
        return newb

    def place_stone(self, coord, color):
        pos = coord_to_index(coord, self.size)
        r, c = divmod(pos, self.size)

        if self.board[r, c] != self.EMPTY:
            return False

        self.board[r, c] = color

        self._remove_captured_groups(3 - color)
        self._remove_captured_groups(color)

        return True

    def _get_group(self, start_pos):
        color = self.board[start_pos // self.size, start_pos % self.size]

        if color == self.EMPTY:
            return [], self.EMPTY

        queue = [start_pos]
        visited = set()
        group = []

        while queue:
            pos = queue.pop()

            if pos in visited:
                continue

            visited.add(pos)

            r, c = divmod(pos, self.size)

            if self.board[r, c] == color:
                group.append(pos)
                queue.extend(get_neighbors(pos, self.size))

        return group, color

    def _count_liberties(self, group):
        liberties = set()

        for pos in group:
            for n in get_neighbors(pos, self.size):
                r, c = divmod(n, self.size)
                if self.board[r, c] == self.EMPTY:
                    liberties.add(n)

        return len(liberties)

    def _remove_captured_groups(self, color):
        visited = set()

        for pos in range(self.size * self.size):
            if pos in visited:
                continue

            r, c = divmod(pos, self.size)

            if self.board[r, c] == color:
                group, _ = self._get_group(pos)
                visited.update(group)

                if self._count_liberties(group) == 0:
                    for g in group:
                        gr, gc = divmod(g, self.size)
                        self.board[gr, gc] = self.EMPTY

    def get_group_liberties(self, coords):
        liberties = set()

        for coord in coords:
            pos = coord_to_index(coord, self.size)

            for n in get_neighbors(pos, self.size):
                r, c = divmod(n, self.size)
                if self.board[r, c] == self.EMPTY:
                    liberties.add(n)

        return len(liberties)

    def render(self):
        symbols = {
            self.EMPTY: '.',
            self.BLACK: 'X',
            self.WHITE: 'O'
        }

        lines = []

        for r in range(self.size - 1, -1, -1):
            row = " ".join(symbols[int(self.board[r, c])] for c in range(self.size))
            lines.append(f"{r+1:2d} {row}")

        lines.append("   " + " ".join(LETTERS[:self.size]))

        return "\n".join(lines)


# =========================================================
# 3. TRAJECTORY GENERATION
# =========================================================


def generate_trajectories(candidate_moves, depth=3):
    """
    Generate tactical trajectories.

    Example:
        depth=3
        (white_move, black_response, white_followup)
    """

    trajectories = list(itertools.product(candidate_moves, repeat=depth))

    return trajectories


# =========================================================
# 4. QUANTUM BORN MACHINE
# =========================================================

class BornTrajectoryCircuit:

    def __init__(self, num_states):

        self.num_qubits = int(np.ceil(np.log2(max(num_states, 2))))

        self.circuit = RealAmplitudes(
            self.num_qubits,
            reps=3,
            entanglement='full'
        )

    @property
    def num_parameters(self):
        return self.circuit.num_parameters

    def statevector(self, params):
        return Statevector.from_instruction(
            self.circuit.assign_parameters(params)
        )


# =========================================================
# 5. TRAJECTORY EVALUATOR
# =========================================================

class TrajectoryEvaluator:

    def __init__(self, board, group, candidate_moves, depth=3):

        self.board = board
        self.group = group
        self.candidate_moves = candidate_moves

        self.trajectories = generate_trajectories(candidate_moves, depth)

        self.num_trajectories = len(self.trajectories)

        self.born = BornTrajectoryCircuit(self.num_trajectories)

        print(f"Generated {self.num_trajectories} trajectories")
        print(f"Using {self.born.num_qubits} qubits")

    # -----------------------------------------------------
    # Tactical simulation
    # -----------------------------------------------------

    def evaluate_trajectory(self, trajectory):

        board = self.board.copy()

        colors = [GoBoard.WHITE, GoBoard.BLACK, GoBoard.WHITE]

        for move, color in zip(trajectory, colors):
            board.place_stone(move, color)

        liberties = board.get_group_liberties(self.group)

        # Tactical heuristic
        # ---------------------------------

        if liberties >= 3:
            return 1.0

        elif liberties == 2:
            return 0.7

        elif liberties == 1:
            return 0.2

        return 0.0

    # -----------------------------------------------------
    # Born expectation value
    # -----------------------------------------------------

    def expectation(self, params):

        sv = self.born.statevector(params)

        probs = np.abs(sv.data) ** 2

        total = 0.0

        for idx, trajectory in enumerate(self.trajectories):

            if idx >= len(probs):
                continue

            p = probs[idx]

            if p < 1e-8:
                continue

            score = self.evaluate_trajectory(trajectory)

            total += p * score

        return total

    # -----------------------------------------------------
    # Aggregate move probabilities
    # -----------------------------------------------------

    def move_distribution(self, params):

        sv = self.born.statevector(params)

        probs = np.abs(sv.data) ** 2

        move_probs = {m: 0.0 for m in self.candidate_moves}

        for idx, trajectory in enumerate(self.trajectories):

            if idx >= len(probs):
                continue

            first_move = trajectory[0]

            move_probs[first_move] += probs[idx]

        return move_probs


# =========================================================
# 6. OPTIMIZER
# =========================================================

class Optimizer:

    def __init__(self, objective, nparams):
        self.objective = objective
        self.nparams = nparams

    def optimize(self, theta, steps=100, lr=0.15, eps=1e-2):
    
        history = []

        for step in range(steps):

            value = self.objective(theta)
            history.append(value)

            grad = np.zeros_like(theta)

            for i in range(len(theta)):

                plus = theta.copy()
                minus = theta.copy()

                plus[i] += eps
                minus[i] -= eps

                grad[i] = (
                    self.objective(plus)
                    - self.objective(minus)
                ) / (2 * eps)

            theta += lr * grad

            if step % 10 == 0:
                print(f"\nSTEP {step}")
                print(f"Expectation = {value:.4f}")
                print(f"Gradient norm = {np.linalg.norm(grad):.4f}")

        return theta, history


# =========================================================
# 7. SETUP
# =========================================================


def setup_problem():

    board = GoBoard(9)

    black = ["F2", "G2", "E2", "H2", "I2"]
    white = ["E3", "F3", "G3", "H3", "I3", "D2"]

    for s in black:
        board.place_stone(s, GoBoard.BLACK)

    for s in white:
        board.place_stone(s, GoBoard.WHITE)

    group = black

    candidates = ["E1", "F1", "G1", "H1", "I1"]

    return board, group, candidates


# =========================================================
# 8. MAIN
# =========================================================


def main():

    start = time.time()

    board, group, candidates = setup_problem()

    print(board.render())

    evaluator = TrajectoryEvaluator(
        board,
        group,
        candidates,
        depth=3
    )

    theta = np.random.uniform(
        0,
        np.pi,
        evaluator.born.num_parameters
    )

    optimizer = Optimizer(
        evaluator.expectation,
        evaluator.born.num_parameters
    )

    theta, history = optimizer.optimize(theta)

    print("\n" + "="*60)
    print("FINAL DISTRIBUTION")
    print("="*60)

    probs = evaluator.move_distribution(theta)

    for move, p in sorted(probs.items(), key=lambda x: -x[1]):
        print(f"{move}: {p:.4f} ({100*p:.2f}%)")

    best = max(probs, key=probs.get)

    print("\nBEST MOVE:", best)

    print(f"\nElapsed: {time.time() - start:.2f}s")


if __name__ == '__main__':
    main()

""""
# Principali differenze rispetto al vecchio modello

## Prima

Distribuzione su:

[
P(a|s)
]

Singole mosse.

---

## Ora

Distribuzione su:

[
P(a_1,a_2,a_3|s)
]

Traiettorie tattiche.

---

# Effetto teorico

Ora il modello:

* penalizza mosse che sembrano buone subito ma perdono dopo;
* apprende linee tattiche;
* rappresenta reading probabilistico;
* è molto più vicino al Go reale.

---

# Limiti attuali

## 1. Esplosione combinatoria

Con:

* 5 mosse
* profondità 3

hai:

[
5^3 = 125
]

traiettorie.

Con profondità 5:

[
5^5 = 3125
]

---

## 2. Tactical evaluator ancora semplice

Può essere migliorato con:

* ladder detection
* eye detection
* ko handling
* local ownership
* rollout search

---

# Insight importante

Questo codice implementa davvero:

> distribution-based tactical reasoning over trajectories.

Non più semplicemente:

> move probability estimation.
"""