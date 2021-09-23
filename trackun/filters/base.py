from abc import abstractmethod


class BayesFilter:
    @abstractmethod
    def init(self):
        pass

    @abstractmethod
    def predict(self, upds_k):
        pass

    @abstractmethod
    def update(self, Z, preds_k):
        pass

    @abstractmethod
    def estimate(self, upds_k):
        pass

    def step(self, Z, upds_k):
        # == Predict ==
        preds_k = self.predict(upds_k)

        # == Update ==
        upds_k = self.update(Z, preds_k)

        return upds_k

    def run(self, Zs):
        # Initialize
        upds_k = self.init()

        # Recursive loop
        ests = []
        for Z in Zs:
            upds_k = self.step(Z, upds_k)
            ests_k = self.estimate(upds_k)
            ests.append(ests_k)

        return ests

    def visualizable_run(self, Zs):
        # Initialize
        upds_k = self.init()

        # Recursive loop
        ests = []
        for Z in Zs:
            upds_k = self.step(Z, upds_k)
            ests_k = self.visualizable_estimate(upds_k)
            ests.append(ests_k)

        return ests


class GMSFilter(BayesFilter):
    pass


class SMCFilter(BayesFilter):
    pass
