import copy
import keras
import numpy as np
from timeit import default_timer

model = keras.models.load_model('/Users/theomettezcalifornie22/Desktop/ECE_4/IA/ProjetV25.01/sudoku_model.h5')

game = '''
          8 0 0 0 0 0 0 0 0
          0 0 3 6 0 0 0 0 0
          0 7 0 0 9 0 2 0 0
          0 5 0 0 0 7 0 0 0
          0 0 0 0 4 5 7 0 0
          0 0 0 1 0 0 0 3 0
          0 0 1 0 0 0 0 6 8
          0 0 8 5 0 0 0 1 0
          0 9 0 0 0 0 4 0 0
      '''

def norm(a):
    return (a / 9) - .5


def denorm(a):
    return (a + .5) * 9


def inference_sudoku(sample):
    '''
        This function solves the sudoku by filling blank positions one by one.
    '''

    feat = copy.copy(sample)

    while True:

        out = model.predict(feat.reshape((1, 9, 9, 1)))
        out = out.squeeze()

        pred = np.argmax(out, axis=1).reshape((9, 9)) + 1
        prob = np.around(np.max(out, axis=1).reshape((9, 9)), 2)

        feat = denorm(feat).reshape((9, 9))
        mask = (feat == 0)

        if mask.sum() == 0:
            break

        prob_new = prob * mask

        ind = np.argmax(prob_new)
        x, y = (ind // 9), (ind % 9)

        val = pred[x][y]
        feat[x][y] = val
        feat = norm(feat)

    return pred


def solve_sudoku(game):
    # Verify if `game` is a string; if so, process it
    if isinstance(game, str):
        game = game.replace('\n', '').replace(' ', '')
        game = np.array([int(j) for j in game]).reshape((9, 9, 1))

    # Normalize and solve
    game = norm(game)
    game = inference_sudoku(game)
    return game


# Start execution
start = default_timer()

# Ensure `game` is only parsed as a string once
game = solve_sudoku(game)  # Resolution of the grid.

print('Solved puzzle:\n')  # Display the solved puzzle
print(game)
np.sum(game, axis=1)  # Sum the rows to verify Sudoku integrity.

# Solve Sudoku again
result = solve_sudoku(game)  # No need for conditional; we already solved.

execution = default_timer() - start
print("Le temps de résolution est de : ", execution * 1000, " ms")
