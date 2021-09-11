class ConstantDetectionModel:
    def __init__(self, pD):
        self.pD = pD

    def get_probability(self, _=None):
        return self.pD
