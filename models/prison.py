"""Three-player Prisoner's Dilemma: payoff table and the Player interface.

Every agent implements play(); agentName() is optional. Actions: 0 = cooperate, 1 = defect.

PAYOFF[a][b][c] is the payoff to the player choosing a when the other two
players choose b and c. Defecting always pays exactly 2 more than cooperating,
and each defecting opponent costs 3.
"""

PAYOFF = [[[6, 3], [3, 0]], [[8, 5], [5, 2]]]


class Player(object):
    def agentName(self):
        return self.__class__.__name__

    def play(self, myHistory, oppHistory1, oppHistory2):
        raise NotImplementedError("play not implemented")
