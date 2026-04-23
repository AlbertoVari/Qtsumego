"""
Quantum Go Life Analyzer - Born Machine Approach
Variational quantum algorithm for evaluating stone group survival in Go
"""


import numpy as np
from typing import List, Dict, Tuple, Optional
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, Operator
from qiskit.circuit.library import RealAmplitudes, TwoLocal
from qiskit.primitives import StatevectorSampler
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# =========================
# 1. COSTANTI E UTILITIES
# =========================

LETTERS = "ABCDEFGHI"
Board = np.ndarray
Move = str
Coordinate = str


def coord_to_index(coord: Coordinate, board_size: int) -> int:
    """Convert Go coordinate (e.g., 'E5') to linear index."""
    col = LETTERS.index(coord[0].upper())
    row = int(coord[1:]) - 1
    return row * board_size + col


def index_to_coord(index: int, board_size: int) -> Coordinate:
    """Convert linear index to Go coordinate."""
    row, col = divmod(index, board_size)
    return f"{LETTERS[col]}{row + 1}"


def get_neighbors(position: int, board_size: int) -> List[int]:
    """Get orthogonal neighbors of a position on the board."""
    row, col = divmod(position, board_size)
    neighbors = []
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nr, nc = row + dr, col + dc
        if 0 <= nr < board_size and 0 <= nc < board_size:
            neighbors.append(nr * board_size + nc)
    return neighbors


# =========================
# 2. LOGICA CLASSICA DEL GO
# =========================

class GoBoard:
    """Classical Go board representation with capture rules."""
    
    EMPTY, BLACK, WHITE = 0, 1, 2
    
    def __init__(self, board_size: int = 9):
        self.size = board_size
        self.board = np.zeros((board_size, board_size), dtype=int)
    
    def place_stone(self, coord: Coordinate, color: int) -> bool:
        """Place a stone and handle captures. Returns True if move is valid."""
        pos = coord_to_index(coord, self.size)
        row, col = divmod(pos, self.size)
        
        if self.board[row, col] != self.EMPTY:
            return False
        
        self.board[row, col] = color
        self._remove_captured_groups(3 - color)  # Remove opponent captures first
        self._remove_captured_groups(color)       # Then check self-capture (ko rule not implemented)
        return True
    
    def _get_group(self, start_pos: int) -> Tuple[List[int], int]:
        """BFS to find connected group and its color."""
        color = self.board[start_pos // self.size, start_pos % self.size]
        if color == self.EMPTY:
            return [], self.EMPTY
        
        group, visited, queue = [], set(), [start_pos]
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
    
    def _count_liberties(self, group: List[int]) -> int:
        """Count unique liberties of a group."""
        liberties = set()
        for pos in group:
            for neighbor in get_neighbors(pos, self.size):
                r, c = divmod(neighbor, self.size)
                if self.board[r, c] == self.EMPTY:
                    liberties.add(neighbor)
        return len(liberties)
    
    def _remove_captured_groups(self, color: int):
        """Remove all groups of given color with zero liberties."""
        visited = set()
        for pos in range(self.size * self.size):
            if pos in visited:
                continue
            r, c = divmod(pos, self.size)
            if self.board[r, c] == color:
                group, _ = self._get_group(pos)
                visited.update(group)
                if self._count_liberties(group) == 0:
                    for g_pos in group:
                        gr, gc = divmod(g_pos, self.size)
                        self.board[gr, gc] = self.EMPTY
    
    def get_liberties_of_group(self, coords: List[Coordinate]) -> int:
        """Count liberties of a specific group given by coordinates."""
        group_positions = [coord_to_index(c, self.size) for c in coords]
        liberties = set()
        for pos in group_positions:
            for neighbor in get_neighbors(pos, self.size):
                r, c = divmod(neighbor, self.size)
                if self.board[r, c] == self.EMPTY:
                    liberties.add(neighbor)
        return len(liberties)
    
    def copy(self) -> 'GoBoard':
        """Return a deep copy of the board."""
        new_board = GoBoard(self.size)
        new_board.board = self.board.copy()
        return new_board


# =========================
# 3. QUANTUM COMPONENTS
# =========================

class BornMachineCircuit:
    """Parameterized quantum circuit for Born machine probability distribution."""
    
    def __init__(self, num_qubits: int, reps: int = 2, entanglement: str = 'linear'):
        """
        Initialize ansatz circuit.
        
        Args:
            num_qubits: Number of qubits (determines move space size)
            reps: Number of repetition layers in the ansatz
            entanglement: Entanglement strategy ('linear', 'full', 'circular')
        """
        self.num_qubits = num_qubits
        # Use hardware-efficient ansatz with RY rotations + CX entanglement
        self.circuit = RealAmplitudes(
            num_qubits=num_qubits,
            reps=reps,
            entanglement=entanglement,
            insert_barriers=False
        )
    
    def build_statevector(self, parameters: np.ndarray) -> Statevector:
        """Build statevector from parameters."""
        return Statevector.from_instruction(self.circuit.assign_parameters(parameters))
    
    @property
    def num_parameters(self) -> int:
        return self.circuit.num_parameters


class QuantumLifeEvaluator:
    """Variational quantum algorithm for Go group life evaluation."""
    
    def __init__(self, board: GoBoard, group_coords: List[Coordinate], 
                 candidate_moves: List[Coordinate]):
        """
        Initialize the quantum evaluator.
        
        Args:
            board: Current Go board state
            group_coords: Coordinates of the group to analyze
            candidate_moves: List of candidate moves to consider
        """
        self.board = board
        self.group = group_coords
        self.candidate_moves = candidate_moves
        self.num_moves = len(candidate_moves)
        
        # Calculate required qubits: ceil(log2(num_moves))
        self.num_qubits = int(np.ceil(np.log2(max(self.num_moves, 2))))
        
        # Initialize quantum components
        self.ansatz = BornMachineCircuit(self.num_qubits)
        self.sampler = StatevectorSampler()
        
        logger.info(f"Initialized with {self.num_qubits} qubits for {self.num_moves} moves")
    
    def _compute_life_expectation(self, parameters: np.ndarray) -> float:
        """
        Compute expected life value using Born rule.
        
        The expectation value is: E[life] = Σᵢ |⟨i|ψ(θ)⟩|² · life(moveᵢ)
        """
        # Get quantum probability distribution
        statevector = self.ansatz.build_statevector(parameters)
        probabilities = np.abs(statevector.data) ** 2
        
        # Compute weighted expectation over valid moves
        life_expectation = 0.0
        for idx in range(min(self.num_moves, len(probabilities))):
            if probabilities[idx] < 1e-10:
                continue
            
            # Simulate move classically
            board_after = self.board.copy()
            board_after.place_stone(self.candidate_moves[idx], GoBoard.WHITE)
            
            # Evaluate life value (1.0 = alive, 0.0 = captured)
            life_value = float(board_after.get_liberties_of_group(self.group) > 0)
            life_expectation += probabilities[idx] * life_value
        
        return life_expectation
    
    def evaluate(self, parameters: np.ndarray) -> float:
        """Public method to evaluate life expectation."""
        return self._compute_life_expectation(parameters)
    
    def get_probabilities(self, parameters: np.ndarray) -> Dict[Move, float]:
        """Get probability distribution over candidate moves."""
        statevector = self.ansatz.build_statevector(parameters)
        probs = np.abs(statevector.data) ** 2
        
        return {
            move: float(probs[idx]) if idx < len(probs) else 0.0
            for idx, move in enumerate(self.candidate_moves)
        }


# =========================
# 4. OPTIMIZER (GRADIENT-FREE)
# =========================

class SimpleOptimizer:
    """Simple gradient-free optimizer using finite differences."""
    
    def __init__(self, objective_func, num_params: int, 
                 learning_rate: float = 0.3, epsilon: float = 1e-2):
        self.objective = objective_func
        self.num_params = num_params
        self.lr = learning_rate
        self.eps = epsilon
    
    def optimize(self, initial_params: np.ndarray, steps: int = 40, 
                 verbose: bool = True) -> Tuple[np.ndarray, List[float]]:
        """Run optimization loop."""
        params = initial_params.copy()
        history = []
        
        for step in range(steps):
            current_value = self.objective(params)
            history.append(current_value)
            
            # Finite difference gradient estimation
            gradient = np.zeros(self.num_params)
            for i in range(self.num_params):
                params_plus = params.copy()
                params_minus = params.copy()
                params_plus[i] += self.eps
                params_minus[i] -= self.eps
                
                grad_i = (self.objective(params_plus) - self.objective(params_minus)) / (2 * self.eps)
                gradient[i] = grad_i
            
            # Gradient ascent (we want to maximize life expectation)
            params += self.lr * gradient
            
            if verbose and (step % 5 == 0 or step == steps - 1):
                logger.info(f"Step {step:2d}: Life={current_value:.4f}, ‖∇‖={np.linalg.norm(gradient):.4f}")
        
        return params, history


# =========================
# 5. MAIN EXECUTION
# =========================

def setup_problem() -> Tuple[GoBoard, List[Coordinate], List[Coordinate]]:
    """Setup the Go problem from the original specification."""
    problem = {
        "board_size": 9,
        "black_stones": ["E5", "E4", "F4"],
        "white_stones": ["D5", "F5", "E6"],
        "group_to_analyze": ["E5", "E4", "F4"],  # Black group
        "player_to_move": "white"
    }
    
    board = GoBoard(problem["board_size"])
    for coord in problem["black_stones"]:
        board.place_stone(coord, GoBoard.BLACK)
    for coord in problem["white_stones"]:
        board.place_stone(coord, GoBoard.WHITE)
    
    # Generate candidate moves adjacent to the group
    group_positions = [coord_to_index(c, problem["board_size"]) for c in problem["group_to_analyze"]]
    candidate_set = set()
    for pos in group_positions:
        for neighbor in get_neighbors(pos, problem["board_size"]):
            r, c = divmod(neighbor, problem["board_size"])
            if board.board[r, c] == GoBoard.EMPTY:
                candidate_set.add(index_to_coord(neighbor, problem["board_size"]))
    
    return board, problem["group_to_analyze"], sorted(list(candidate_set))


def main():
    """Main execution function."""
    logger.info("🎮 Quantum Go Life Analyzer - Starting")
    
    # Setup problem
    board, group, candidates = setup_problem()
    logger.info(f"Group to analyze: {group}")
    logger.info(f"Candidate moves: {candidates}")
    
    # Initialize quantum evaluator
    evaluator = QuantumLifeEvaluator(board, group, candidates)
    
    # Initialize optimizer
    initial_theta = np.random.uniform(0, np.pi, evaluator.ansatz.num_parameters)
    optimizer = SimpleOptimizer(evaluator.evaluate, evaluator.ansatz.num_parameters)
    
    # Run optimization
    logger.info("🔄 Starting variational optimization...")
    optimal_params, history = optimizer.optimize(initial_theta, steps=40)
    
    # Results
    final_value = evaluator.evaluate(optimal_params)
    probabilities = evaluator.get_probabilities(optimal_params)
    
    print("\n" + "="*60)
    print("📊 RESULTS")
    print("="*60)
    print(f"Final expected life value: {final_value:.4f}")
    print(f"\nMove probability distribution:")
    for move, prob in sorted(probabilities.items(), key=lambda x: -x[1]):
        print(f"  {move}: {prob:.4f} ({prob*100:.2f}%)")
    print(f"\n🎯 Most promising move: {max(probabilities, key=probabilities.get)}")
    
    return optimal_params, probabilities, history


if __name__ == "__main__":
    main()