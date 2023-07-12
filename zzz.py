from typing import Any


class ppp:
    def __init__(self, para1) -> None:
        self.para1 = para1
        self.func = lambda x: x + para1
    
    def __call__(self, y) -> Any:
        return self.func(y)

ui = ppp(1)(4)
print(ui)
'''
5
'''

class ttt:
    def __init__(self, pa) -> None:
        self.pa = pa
    
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        print(self.pa)
        return True
     
si = ttt(1)()
print(si)
'''
1
True
'''
        