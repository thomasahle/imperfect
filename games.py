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
        ''' Get the score in {-1,1} relative to x-player '''
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


class NumberGame(WinTieLoseGame):
    def __init__(self, ns, random_hidden):
        super().__init__(random_hidden)
        self.ns = ns

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


class TicTacToe(WinTieLoseGame):
    ...
