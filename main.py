import itertools
import sys
import fractions
import time
import argparse
import numpy as np
import pickle

import algo
import importlib
import games


def sfrac(val):
    return str(fractions.Fraction.from_float(val).limit_denominator())

def print_strat(game, ps, strategy_player=0):
    for h, xv in sorted(ps.items()):
        if h == (): continue
        print('  '*len(h),
              game.repr_priv(h[0]),
              game.repr_hist(h[1:]),
              end=' ')
        if np.isclose(xv, 0):
            print()
        elif game.get_player(h[1:]) == strategy_player:
            print()
        else:
            print('wp', sfrac(xv / ps[h[:-1]]))

def solve(args):
    game_cls = getattr(games, args.game)
    game = game_cls(args)
    ps, vs = algo.solve_game(game, verbose=args.verbose)
    lam = vs[()]

    print()
    print('If strat_player=0')
    print('Value:', sfrac(lam))
    if args.print_strategy:
        print_strat(game, ps)

    if args.save_strategy is not None:
        vals = {h: xv for h, xv in ps.items()}
        with open(args.save_strategy, 'wb') as file:
            pickle.dump(vals, file)

    if args.reverse:
        print()
        print('Solving reverse game', file=sys.stderr)
        ps2, vs2 = algo.solve_game(game, strategy_player=1, verbose=True)
        lam2 = vs2[()]

        print()
        print('If strat_player=1')
        print('Value:', sfrac(-lam2))
        if args.print_strategy:
            print_strat(game, ps2, strategy_player=1)

        if args.save_strategy is not None:
            vals = {h: xv for h, xv in ps2.items()}
            path = args.save_strategy + '.reverse'
            with open(path, 'wb') as file:
                pickle.dump(vals, file)



def quick_solve(game, strategy_player=0):
    ps, vs = init_solver(game, strategy_player)
    status = solver.Solve()
    if status != solver.OPTIMAL:
        print('Didn\'t find optimal solution!')
        print('Status:', status, file=sys.stderr)
        print('Value:', lam)
        return
    val = vs[()]
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
    parser.add_argument('--verbose', action='store_true', help='Verbose')
    subparsers = parser.add_subparsers(required=True)

    parser_solve = subparsers.add_parser('solve')
    parser_solve.add_argument('game', type=str, help='Game to play, like numbers or tictactoe')
    parser_solve.add_argument('--print-strategy', action='store_true', help='Whether to output the strategy (rather than just the result)')
    parser_solve.add_argument('--save-strategy', type=str, help='Output path for LP variables')

    parser_solve.add_argument('--numbers', type=int, default=1, nargs='+')

    parser_solve.add_argument('--a-dice', type=int, default=1)
    parser_solve.add_argument('--b-dice', type=int, default=1)
    parser_solve.add_argument('--sides', type=int, default=6)
    parser_solve.add_argument('--joker', action='store_true')
    parser_solve.set_defaults(func=solve)

    parser_plot = subparsers.add_parser('plot')
    parser_plot.set_defaults(func=plot)
    parser_plot.add_argument('--rounds', type=int, default=2)
    parser_plot.add_argument('--max-a', type=int, default=10)
    parser_plot.add_argument('--max-b', type=int, default=10)

    args = parser.parse_args()
    args.func(args)

