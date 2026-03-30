print("Starting script...")
import pandas as pd
print("Pandas imported")
import numpy as np
print("Numpy imported")
import matplotlib
matplotlib.use('Agg')
print("Matplotlib backend set")
import matplotlib.pyplot as plt
print("Matplotlib imported")

print("Creating simple plot...")
plt.figure()
plt.plot([1, 2, 3], [1, 2, 3])
plt.savefig('test_plot.png')
print("Plot saved as test_plot.png")
print("Script completed!")
