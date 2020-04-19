import numpy as np


class MAP():
    def __init__(self):
        self.n_data = 0
        self.total_ap = 0

    def update(self, predicts, ans):
        self.n_data += 1
        ap = 0
        cnt = 0
        for idx, predict in enumerate(predicts):
            if predict.lower() in ans:
                cnt += 1
                ap += cnt / (idx+1)
        ap /= len(ans)
        print(ap)
        self.total_ap += ap

    @property
    def score(self):
        return self.total_ap / self.n_data
