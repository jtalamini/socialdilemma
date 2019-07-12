
class Prisoner():

    def __init__(self):
        self.payoff = [
            [[2,2],[10,0]],
            [[0,10],[5,5]]
        ]

    def play(self, actions):
        return self.payoff[actions[0]][actions[1]]
