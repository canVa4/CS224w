import numpy as np

pot12 = np.array([[1, 0.9], [0.9, 1]])
pot34 = np.array([[1, 0.9], [0.9, 1]])
pot23 = np.array([[0.1, 1], [1, 0.1]])
pot35 = np.array([[0.1, 1], [1, 0.1]])
prior2 = np.array([[1, 0.1], [0.1, 1]])
prior4 = np.array([[1, 0.1], [0.1, 1]])

b1 = prior2 @ pot12 @ pot23 @ (pot35.sum(axis=0) + prior4 @ pot34.sum(axis=0))
print(b1)
