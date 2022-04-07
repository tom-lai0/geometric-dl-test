import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns
import torch
import random
import plotly.express as px
import plotly.graph_objects as go
from torch.utils.data import Dataset, DataLoader


class PointData:
    def __init__(self, data):
        self.data = data.float()
        self.num_data = len(data)
        self.shape = data.shape

    def calc_dist(self, p=1):
        # return matrix of squared pairwise l2 norm
        c = (
            (self.data ** 2).sum(axis=1).reshape(self.num_data, 1) *
             torch.ones((self.num_data, self.num_data))
        )
        a = self.data @ self.data.T
        return c.T + c - 2 * a

    def plot(self, direction=0):
        # direction:
        #  x: 0, y: 1, z: 2
        if direction in {0, 1, 2}:
            # fig = go.Figure(data=[go.Scatter3d(
            #     x=self.data[:, 0], y=self.data[:, 1], z=self.data[:, 2],
            #     mode='markers',
            #     marker=dict(
            #         size=5,
            #         color=self.data[:, direction],
            #         colorscale='Viridis',
            #         opacity=0.8
            #     )
            # )])
            # fig.show()
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            ax.scatter(
                self.data[:, 0],
                self.data[:, 1],
                self.data[:, 2],
                c = self.data[:, direction]
            )
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel('z')
            plt.xlim(-1.4, 1.4)
            plt.ylim(-1.4, 1.4)
            plt.show()

        else:
            print('PointData: plot: Invalid direction')


class DataGenerator:
    def __init__(self, shape=1, length=1):
        self.shape = shape  # 1 for sphere
        self.length = length  # radius for sphere

    def set_shape(self, shape):
        self.shape = shape

    def set_length(self, length):
        self.length = length

    def generate(self, num_data=1000):
        if self.shape == 1:
            phi = torch.rand((num_data, 1)) * 2 * np.pi
            theta = torch.rand((num_data, 1)) * np.pi
            x = torch.cat(
                [
                    torch.cos(phi) * torch.sin(theta),
                    torch.sin(phi) * torch.sin(theta),
                    torch.cos(theta)
                ],
                dim=1
            )
            return PointData(x)

        # elif self.shape == 2:
        #     pass

        else:
            print('DataGenerator: generate: Invalid shape')


class PointDataset(Dataset):
    def __init__(self, point_data):
        self.data = point_data.data
        self.shape = point_data.shape

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)



