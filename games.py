from collections import Counter
import itertools


################################################################
# Game definition
################################################################

class Game:
    def __init__(self, random_hidden):
        self.root = ()
        self.random_hidden = random_hidden

    def privates(self, player):
        raise NotImplementedError

    def children(self, h):
        raise NotImplementedError

    def get_player(self, h):
        # Next player to move
        return len(h) % 2

    def is_leaf(self, h):
        return next(self.children(h), None) is None

    def histories(self, root=()):
        yield root
        for c in self.children(root):
            yield from self.histories(root=c)

    def score(self, priv1, priv2, hist):
        """ The score of the leaf node relative to player 1 """
        raise NotImplementedError

    def repr_hist(self, hist):
        if not hist:
            return '()'
        return "(" + ",".join(map(str, hist)) + ")"


WIN, TIE, LOSE = range(3)

class WinTieLoseGame(Game):
    def privates(self, player):
        return ((r,) for r in (WIN, TIE, LOSE))

    def winner(self, h):
        # Returns 0, 1 or None for a tie
        raise NotImplementedError

    def score(self, priv1, priv2, hist):
        ''' Get the score in {-1,0,1} relative to x-player '''
        d1, = priv1
        d2, = priv2
        winner = self.winner(hist)
        res = 0
        if d1 == WIN and winner == 0 \
                or d1 == TIE and winner is None \
                or d1 == LOSE and winner == 1:
            res += 1
        if d2 == WIN and winner == 1 \
                or d2 == TIE and winner is None \
                or d2 == LOSE and winner == 0:
            res -= 1
        return res


class numbers(WinTieLoseGame):
    def __init__(self, args):
        super().__init__(args.random)
        self.ns = [n+1 for n in args.numbers]

    def children(self, h):
        if len(h) >= len(self.ns):
            return
        for i in range(self.ns[len(h)]):
            yield h + (i,)

    def winner(self, h):
        a = sum(h[::2])
        b = sum(h[1::2])
        if a > b: return 0
        if a < b: return 1
        return None


ttt_syms = (
    (0, 1, 2, 3, 4, 5, 6, 7, 8),
    (2, 5, 8, 1, 4, 7, 0, 3, 6),
    (8, 7, 6, 5, 4, 3, 2, 1, 0),
    (6, 3, 0, 7, 4, 1, 8, 5, 2),
    (2, 1, 0, 5, 4, 3, 8, 7, 6),
    (8, 5, 2, 7, 4, 1, 6, 3, 0),
    (6, 7, 8, 3, 4, 5, 0, 1, 2),
    (0, 3, 6, 1, 4, 7, 2, 5, 8),
)
ttt_goals = (
    (0, 1, 2),
    (3, 4, 5),
    (6, 7, 8),
    (0, 3, 6),
    (1, 4, 7),
    (2, 5, 8),
    (0, 4, 8),
    (2, 4, 6),
)
class tictactoe(WinTieLoseGame):
    def __init__(self, args):
        super().__init__(args.random)

    def _toboard(self, h):
        board = [-1] * 9
        for i in h[::2]:
            board[i] = 0
        for i in h[1::2]:
            board[i] = 1
        return board

    def winner(self, h):
        board = self._toboard(h)
        for goal in ttt_goals:
            if all(board[i] == 0 for i in goal):
                return 0
            if all(board[i] == 1 for i in goal):
                return 1
        return None

    def _sym(self, h):
        # Get smallest symmetric representative of history
        return min(tuple(sym[i] for i in sym) for sym in ttt_syms)

    def children(self, h):
        if self.winner(h) is not None or len(h) == 9:
            return
        opts = set()
        for i, b in enumerate(self._toboard(h)):
            if b == -1:
                opts.add(self._sym(h + (i,)))
        yield from opts


dudo_doubt = None

class dudo(Game):
    def __init__(self, args):
        super().__init__(args.random)
        self.dice = [args.a_dice, args.b_dice]
        self.sides = args.sides
        self.joker = args.joker
        self.max = sum(self.dice) * (2 if self.joker else 1)

    def privates(self, player):
        # Should remove symmetries here, but for random mode (default for dudo)
        # we would need a weighted sample, which is currently not supported
        return itertools.product(range(1, self.sides+1), repeat=self.dice[player])

    def children(self, h):
        if h and h[-1] is dudo_doubt:
            return
        n, d = (1, 0) if len(h) == 0 else h[-1]
        for n1 in range(n, self.max+1):
            for d1 in range(1, self.sides+1):
                if (n1, d1) > (n, d):
                    yield h + ((n1, d1),)
        if len(h) > 0:
            yield h + (dudo_doubt,)

    def score(self, priv1, priv2, hist):
        assert hist[-1] == dudo_doubt
        assert len(hist) >= 2
        n, d = hist[-2]
        cnt = Counter(priv1 + priv2)
        # The player that made the call (not the doubter)
        player = self.get_player(hist)
        # The call was true
        if cnt[n] >= d:
            return 1 if player == 0 else -1
        else:
            return -1 if player == 0 else 1

