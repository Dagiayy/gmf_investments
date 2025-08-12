import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import src.models as m

print(dir(m))
print(m.load_series)  # This should print <function load_series at ...>
