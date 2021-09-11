class ConstantSurvivalModel:
    def __init__(self, pS):
        self.pS = pS

    def get_probability(self, _=None):
        return self.pS
