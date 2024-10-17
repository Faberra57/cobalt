import sys
import os
import tool_kit as tk
import numpy as np
import pandas as pd

print("Répertoire actuel :", os.getcwd())
path_to_tool_kit = os.path.abspath('..')
if path_to_tool_kit not in sys.path:
    sys.path.append(path_to_tool_kit)

