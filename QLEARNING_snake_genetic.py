from QLEARNING import QNetwork
from QLEARNING import QLearning
import numpy as np
from gameModule import ( 
    RIGHT,
    LEFT,
    DOWN,
    UP,
    SNAKE_CHAR,
    FOOD_CHAR,
    WALL_CHAR, 
)

class Snake_q:
    def __init__(self,game):
        self.game = game
        self.hunger = 100 
        print(type(self.hunger))
        self.maxHunger = 100
        self.nbrMove = 0
        self.previous_moves = []
        self.q_network = QNetwork(26, 4)  # 26 entrées (vision) et 4 sorties (mouvements)
        self.q_learning = QLearning(self.q_network)

    def choose_next_move(self, state):
        vision = self.get_simplified_state(state)
        if self.hunger > 0:
            self.hunger -= 1
            self.nbrMove += 1
            action = self.q_learning.choose_action(vision)
            MOVEMENT = (RIGHT, LEFT, UP, DOWN)
            self.previous_moves.append(MOVEMENT[action])
            if len(self.previous_moves) >= 3:
                self.previous_moves.pop(0)

            # Extraire les informations nécessaires
            # Obtenir l'état suivant en appelant la méthode get_next_state
            #next_state, reward, done, _ = self.game.next_tick(action)

            #self.q_learning.learn(state, action, reward, next_state)  # Apprentissage ajouté ici

            return MOVEMENT[action]
        return "starve"

    def learn(self, state, action, reward, next_state):
        self.q_learning.learn(state, action, reward, next_state)

    def get_nbr_move(self):
        """
        Returns the number of moves done by the snake from the beginning of the game (int)
        """
        return self.nbrMove    

    def get_simplified_state(self, state):
        """
        returns a matrix of elements surrounding the snake and the preivous two
        moves, this serves as the input for the neural network.
        """
        res = self.get_line_elem(RIGHT, state)
        res += self.get_line_elem((DOWN[0], RIGHT[1]), state)
        res += self.get_line_elem(DOWN, state)
        res += self.get_line_elem((DOWN[0], LEFT[1]), state)
        res += self.get_line_elem(LEFT, state)
        res += self.get_line_elem((UP[0], LEFT[1]), state)
        res += self.get_line_elem(UP, state)
        res += self.get_line_elem((UP[0], RIGHT[1]), state)

        if len(self.previous_moves) == 0:
            res += [0, 0]
        elif len(self.previous_moves) == 1:  # previous previous move
            res += [
                self.previous_moves[0][0] / 2,
                self.previous_moves[0][1] / 2,
            ]
        else:
            res += [
                self.previous_moves[0][0] + self.previous_moves[1][0] / 2,
                self.previous_moves[0][1] + self.previous_moves[1][1] / 2,
            ]

        # Convertir la liste 1D en tableau 2D avec la forme (1, 12)
        return np.array([res])
    
    '''
    def get_simplified_state(self, state):
        """
        returns a matrix of elements surrounding the snake and the preivous two
        moves, this serves as the input for the neural network.
        """
        res = self.get_line_elem(RIGHT, state)
        res += self.get_line_elem((DOWN[0], RIGHT[1]), state)
        res += self.get_line_elem(DOWN, state)
        res += self.get_line_elem((DOWN[0], LEFT[1]), state)
        res += self.get_line_elem(LEFT, state)
        res += self.get_line_elem((UP[0], LEFT[1]), state)
        res += self.get_line_elem(UP, state)
        res += self.get_line_elem((UP[0], RIGHT[1]), state)

        if len(self.previous_moves) == 0:
            res += [0, 0]
        elif len(self.previous_moves) == 1:  # previous previous move
            res += [
                self.previous_moves[0][0] / 2,
                self.previous_moves[0][1] / 2,
            ]
        else:
            res += [
                self.previous_moves[0][0] + self.previous_moves[1][0] / 2,
                self.previous_moves[0][1] + self.previous_moves[1][1] / 2,
            ]

        return res
    '''

    def get_line_elem(self, direction, state):
        """
        returns a list of all elements in a straight line in a certain direction
        from the head of the snake
        """
        grid, score, alive, snake = state
        res = [0, 0, 0]  # food, snake, wall
        current = (snake[0][0] + direction[0], snake[0][1] + direction[1])
        distance = 1  # Distance between the snake head and current position

        while self.is_in_grid(current, grid) and 0 in res:
            if FOOD_CHAR == grid[current[0]][current[1]]:
                res[0] = 1 / distance
            elif not res[1] and SNAKE_CHAR == grid[current[0]][current[1]]:
                res[1] = 1 / distance
            elif not res[2] and WALL_CHAR == grid[current[0]][current[1]]:
                res[2] = 1 / distance

            current = (current[0] + direction[0], current[1] + direction[1])
            distance += 1

        # For the border of the board (!= WALL_CHAR)
        if res[2] == 0:
            res[2] = 1 / distance

        return res
    
    def is_in_grid(self, pos, grid):
        """
        Checks if an element is in the grid
        """
        return 0 <= pos[0] < len(grid) and 0 <= pos[1] < len(grid[0])

    def eat(self):
        """
        Increase the hunger of the snake. This hunger cannot exceed a certain value.
        """
        self.hunger += 75

        if self.hunger > 500:
            self.hunger = 500

    def reset_state(self):
        """
        Restore the hunger of the snake and reset its number of moves.
        """
        self.hunger = self.maxHunger
        self.nbrMove = 0