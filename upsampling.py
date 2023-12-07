import numpy as np
import torch
from imblearn.combine import SMOTEENN

x = np.array(torch.tensor(np.load('./data/final/x.npy'), dtype=torch.float32))
y = np.array(torch.tensor(np.load('./data/final/y.npy'), dtype=torch.long))

sm = SMOTEENN(random_state=598)
x_os, y_os = sm.fit_resample(x, y)

print(x_os.shape[0], y_os.sum())

np.save('./data/final/x_os.npy', x_os)
np.save('./data/final/y_os.npy', y_os)

