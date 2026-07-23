import os
import sys

# 让 tests/ 能 import 仓库根目录下的模块(heartbeat / synth / classifier / ...)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
