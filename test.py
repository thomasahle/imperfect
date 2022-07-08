import games
import algo

def test_ttt():
    game = games.tictactoe()
    algo.solve_game(game, strategy_player=0, verbose=False):
