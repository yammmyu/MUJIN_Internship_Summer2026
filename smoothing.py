import numpy as np


class ExponentialSmoother:
    """
    实时平滑，每次只处理新来的一个动作
    适合推理时的流式输出
    """

    def __init__(self, alpha=0.3):
        """
        alpha: 平滑因子，0-1之间
               越大响应越快，越小越平滑
        """
        self.alpha = alpha
        self.prev_smoothed = None

    def set_alpha(self, new_alpha):
        if self.alpha != new_alpha:
            print(f"update alpha from {self.alpha} to {new_alpha}")
            self.alpha = new_alpha

    def clear(self):
        self.prev_smoothed = None

    def smooth(self, action):
        """
        action: 当前要执行的动作 (8,)
        """
        if isinstance(action, list):
            action = np.array(action)
        assert isinstance(action, np.ndarray)
        if self.prev_smoothed is None:
            self.prev_smoothed = action
            return action

        # 指数移动平均
        smoothed = self.alpha * action + (1 - self.alpha) * self.prev_smoothed
        self.prev_smoothed = smoothed
        return smoothed
