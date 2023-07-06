from typing import Any


class ppp:
    def __init__(self, para1) -> None:
        self.para1 = para1
        self.func = lambda x: x + para1
    
    def __call__(self, y) -> Any:
        return self.func(y)

ui = ppp(1)(4)

print(ui)
        