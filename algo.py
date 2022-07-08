import time, sys
from ortools.linear_solver import pywraplp


def solve_game(game, strategy_player=0, verbose=False):
    if verbose:
        start = time.time()
        print('Creating solver')

    solver = pywraplp.Solver('', pywraplp.Solver.GLOP_LINEAR_PROGRAMMING)

    if verbose:
        print(f'Took {time.time() - start:0f} seconds')
        start = time.time()
        print('Creating variables')

    ps = {priv + h: solver.NumVar(0, 1, '')
            for priv in game.privates(strategy_player)
            for h in game.histories()}
    ps[()] = solver.NumVar(0, 1, '')

    vs = {priv + h: solver.NumVar(-solver.infinity(), solver.infinity(), '')
            for priv in game.privates(1-strategy_player)
            for h in game.histories()}
    vs[()] = solver.NumVar(-solver.infinity(), solver.infinity(), '')


    if verbose:
        print(f'Took {time.time() - start:.0f} seconds')
        start = time.time()
        print('Adding equality constraints')

    # Constraints for the private part of the game
    solver.Add(ps[()] == 1)
    if game.random_hidden:
        n = sum(1 for priv in game.privates(player=strategy_player))
        for priv in game.privates(player=strategy_player):
            solver.Add(ps[priv] == 1/n)
    else:
        solver.Add(ps[()] == sum(ps[priv] for priv in game.privates(player=strategy_player)))

    # Constraints for the public part of the game
    for priv in game.privates(player=strategy_player):
        for h in game.histories():
            if game.is_leaf(h):
                continue
            # The children of a strategy node has an updated probability
            # summing to their parent. This illustrates us splitting
            # options.
            if game.get_player(h) == strategy_player:
                solver.Add(ps[priv+h]
                        == sum(ps[priv+c] for c in game.children(h)))
            # If the deterministic player is next, the probabilities
            # don't update. The children are all equal to the parent.
            else:
                for c in game.children(h):
                    solver.Add(ps[priv + h] == ps[priv + c])


    if verbose:
        print(f'Took {time.time() - start:.0f} seconds')
        start = time.time()
        print('Adding inequality constraints')

    solver.Maximize(vs[()])

    if game.random_hidden:
        n = sum(1 for priv in game.privates(player=1-strategy_player))
        solver.Add(vs[()] == sum(vs[priv] / n for priv in game.privates(player=1-strategy_player)))
    else:
        for priv in game.privates(player=1-strategy_player):
            solver.Add(vs[()] <= vs[priv])

    for priv2 in game.privates(player=1-strategy_player):
        for h in game.histories():
            if game.is_leaf(h):
                # Evaluate!
                # Note: Input to ps is (start_p private, public)
                con = solver.Constraint(0, 0)
                con.SetCoefficient(vs[priv2 + h], -1)
                for priv1 in game.privates(player=strategy_player):
                    # The score2 function takes as input the private
                    # state of player-1 and player-2, so we have to
                    # remember which one we are.
                    if strategy_player == 0:
                        s = game.score(priv1, priv2, h)
                    # We are always maximizing with respect to the strategy
                    # player. So we might have to invert the score.
                    if strategy_player == 1:
                        s = -game.score(priv2, priv1, h)

                    con.SetCoefficient(ps[priv1 + h], s)
                continue
            # If the strategy player is playing, the value will
            # simply follow the expectation over the distribution.
            if game.get_player(h) == strategy_player:
                solver.Add(vs[priv2 + h]
                        == sum(vs[priv2 + c] for c in game.children(h)))
            # If the deterministic player is playing, they will choose
            # the best move, meaning the parent's value will be smaller
            # (or equal) to that of any child
            else:
                for c in game.children(h):
                    solver.Add(vs[priv2 + h] <= vs[priv2 + c])

    if verbose:
        print(f'Took {time.time() - start:.0f} seconds')
        start = time.time()
        print('Solving', file=sys.stderr)

    status = solver.Solve()

    if verbose:
        print(f'Took {time.time() - start:.0f} seconds')

    if status != solver.OPTIMAL:
        print('Warning: status:', status, file=sys.stderr)
        print(lam.solution_value())

    p_sol = {k: v.solution_value() for k, v in ps.items()}
    v_sol = {k: v.solution_value() for k, v in vs.items()}
    return p_sol, v_sol

