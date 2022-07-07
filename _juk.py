import itertools
import sys
from ortools.linear_solver import pywraplp
import fractions
import time
import argparse
import numpy as np
import pickle

from games import NumberGame


################################################################
# Write as matrix
################################################################

def init_solver(game, strategy_player=0, verbose=False):
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
    ps[game.root] = solver.NumVar(0, 1, '')

    vs = {priv + h: solver.NumVar(-1, 1, '')
            for priv in game.privates(1-strategy_player)
            for h in game.histories()}
    vs[game.root] = solver.NumVar(-1, 1, '')


    if verbose:
        print(f'Took {time.time() - start:.0f} seconds')
        start = time.time()
        print('Adding equality constraints')

    # Constraints for the private part of the game
    solver.Add(ps[game.root] == 1)
    if game.random_hidden:
        n = sum(1 for priv in game.privates(player=strategy_player))
        for priv in game.privates(player=strategy_player):
            solver.Add(ps[priv + game.root] == 1/n)
    else:
        solver.Add(ps[game.root] == sum(ps[priv + game.root] for priv in game.privates(player=strategy_player)))

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

    solver.Maximize(vs[game.root])

    if game.random_hidden:
        n = sum(1 for priv in game.privates(player=1-strategy_player))
        solver.Add(vs[game.root] == sum(vs[priv + game.root] / n for priv in game.privates(player=1-strategy_player)))
    else:
        for priv in game.privates(player=1-strategy_player):
            solver.Add(vs[game.root] <= vs[priv + game.root])

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

    return solver, ps, vs



def sroll(roll):
    assert 0 <= roll <= 2
    if roll == 0: return 'win'
    if roll == 1: return 'tie'
    if roll == 2: return 'lose'

def sfrac(val):
    return str(fractions.Fraction.from_float(val).limit_denominator())



def solve(args):

    if args.tictac:
        if args.sym:
            game = TicTacToeSym(random_hidden=args.random)
        else:
            game = TicTacToeTree(random_hidden=args.random)
    else:
        NS = [int(n)+1 for n in args.numbers]
        game = NumberGame(NS, random_hidden=args.random)

    solver, xvs, vs = init_solver(game, verbose=True)
    lam = vs[game.root]
    t = time.time()
    print('Solving', file=sys.stderr)

    status = solver.Solve()
    if status != solver.OPTIMAL:
        print('Status:', status, file=sys.stderr)
        print(lam.solution_value())
        return

    print(f'Took {time.time()-t:.0f}s', file=sys.stderr)

    print()
    print('If strat_player=0')
    print('Value:', sfrac(lam.solution_value()))
    if args.print_strategy:
        for h, xv in sorted(xvs.items()):
            if h == (): continue
            print('  '*len(h),
                  sroll(h[0]),
                  game.repr_hist(h[1:]),
                  end=' ',
                  )
            if np.isclose(xv.solution_value(), 0):
                print()
            else:
                print('wp', sfrac(xv.solution_value()
                        / xvs[h[:-1]].solution_value()
                        ))

    if args.save_strategy is not None:
        vals = {h: xv.solution_value() for h, xv in xvs.items()}
        with open(args.save_strategy, 'wb') as file:
            pickle.dump(vals, file)

    if args.reverse:
        print()
        print('Solving reverse game', file=sys.stderr)
        solver2, xvs2, vs2 = init_solver(game, strategy_player=1, verbose=True)
        lam2 = vs2[game.root]
        t = time.time()
        print('Solving')
        status = solver2.Solve()
        print(f'Took {time.time()-t:.0f}s', file=sys.stderr)
        assert status == solver.OPTIMAL

        print()
        print('If strat_player=1')
        print('Value:', sfrac(-lam2.solution_value()))
        if args.print_strategy:
            for h, xv in xvs2.items():
                print(sroll(h[0]))
                print(game.repr_hist(h[1:]), sfrac(xv.solution_value()))

        if args.save_strategy is not None:
            vals = {h: xv.solution_value() for h, xv in xvs2.items()}
            path = args.save_strategy + '.reverse'
            with open(path, 'wb') as file:
                pickle.dump(vals, file)



def quick_solve(game, strategy_player=0):
    solver, ps, vs = init_solver(game, strategy_player)
    status = solver.Solve()
    if status != solver.OPTIMAL:
        print('Didn\'t find optimal solution!')
        print('Status:', status, file=sys.stderr)
        print('Value:', lam.solution_value())
        return
    val = vs[()].solution_value()
    if strategy_player == 1:
        return -val
    return val


def plot(args):
    import numpy as np
    import seaborn as sns
    import matplotlib.pylab as plt

    data = np.zeros((args.max_a+1, args.max_b+1))
    for a in range(1, args.max_a+2):
        for b in range(1, args.max_b+2):
            ns = ([a, b] * args.rounds)[:args.rounds]
            print(f'{ns=}, {args.random=}')
            game = NumberGame(ns, random_hidden=args.random)
            val = quick_solve(game)
            print(f'{val=}')
            if args.reverse:
                val1 = quick_solve(game, 1)
                print(f'{val1=}')
                assert np.isclose(val, val1)
            data[a-1, b-1] = val

    labels = np.array([[sfrac(v) for v in row] for row in data])

    ax = sns.heatmap(data,
            linewidth=0.5,
            square=True,
            annot=labels,
            annot_kws=dict(fontsize='xx-small'),
            fmt = '',
            )
    ax.set(xlabel='Second player max number, b',
           ylabel='First player max number, a')
    ax.invert_yaxis()
    form = ' '.join((('[a]', '[b]') * args.rounds)[:args.rounds])
    ax.set_title(f'Expected score of player 1 in\n Game of {form}')

    plt.show()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--random', action='store_true', help='Private choice random')
    parser.add_argument('--reverse', action='store_true', help='Also solve reverse game, just as a check')
    subparsers = parser.add_subparsers()

    parser_solve = subparsers.add_parser('solve')
    parser_solve.add_argument('--tictac', action='store_true', help='Play tictactoe')
    parser_solve.add_argument('--sym', action='store_true', help='Merge symmetric positions')
    parser_solve.add_argument('--numbers', metavar='M', type=int, nargs='+')
    parser_solve.add_argument('--print-strategy', action='store_true', help='Whether to output the strategy (rather than just the result)')
    parser_solve.add_argument('--save-strategy', type=str, help='Output path for LP variables')
    parser_solve.set_defaults(func=solve)

    parser_plot = subparsers.add_parser('plot')
    parser_plot.set_defaults(func=plot)
    parser_plot.add_argument('--rounds', type=int, default=2)
    parser_plot.add_argument('--max-a', type=int, default=10)
    parser_plot.add_argument('--max-b', type=int, default=10)

    args = parser.parse_args()
    args.func(args)

