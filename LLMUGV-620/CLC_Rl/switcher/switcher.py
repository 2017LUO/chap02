# switcher.py
class Switcher:
    """
    轮换技能选择器。
    每创建一个实例，计数器就会从 0 开始。
    默认顺序: ["wait", "wait", 2, 3, 4, 5]
    """

    def __init__(self, order=None):
        # 支持自定义顺序，默认使用题目里的顺序
        self.order = order or ["wait", "wait", 2, 3, 4, 5]
        self.idx = 0

    def next(self):
        """返回当前元素并移动指针。"""
        val = self.order[self.idx]
        self.idx = (self.idx + 1) % len(self.order)
        return val

    # 让实例可以直接像函数那样调用: sw() -> 下一个值
    __call__ = next
