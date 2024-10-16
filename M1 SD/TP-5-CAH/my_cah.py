import numpy as np
import matplotlib.pyplot as plt


def d_min(Di, Dj):
    diff = Di[:, np.newaxis, :] - Dj[np.newaxis, :, :]
    diff = diff.reshape(-1, Di.shape[1])
    return np.min(np.linalg.norm(diff, axis=1))

def d_max(Di, Dj):
    diff = Di[:, np.newaxis, :] - Dj[np.newaxis, :, :]
    diff = diff.reshape(-1, Di.shape[1])
    return np.max(np.linalg.norm(diff, axis=1))

def d_mean(Di, Dj):
    diff = Di[:, np.newaxis, :] - Dj[np.newaxis, :, :]
    diff = diff.reshape(-1, Di.shape[1])
    return np.mean(np.linalg.norm(diff, axis=1))

def d_centroid(Di, Dj):
    return np.linalg.norm(np.mean(Di, axis=0) - np.mean(Dj, axis=0))

if __name__ == '__main__':
    D1 = np.random.rand(5, 2)
    D2 = np.random.rand(7, 2)
    print(d_centroid(D1, D2))