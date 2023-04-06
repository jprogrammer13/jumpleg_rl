import ReplayBuffer

import matplotlib.pyplot as plt
import matplotlib
import joblib
import os
import numpy as np
from scipy import interpolate

target_e = True
filter = True
merged = False

if target_e:
    max_error = .5
    err_treshold = .1
else:  # percentage
    max_error = 1.0
    err_treshold = .15

cmap = matplotlib.cm.jet
norm = matplotlib.colors.Normalize(vmin=0, vmax=max_error)

cmap2 = matplotlib.cm.cool
norm2 = matplotlib.colors.Normalize(vmin=0, vmax=err_treshold)

rb = joblib.load('../runs_small/test/model_latest/ReplayBuffer_test.joblib')
e = rb.mem_size


com0_points = rb.state[:e][:, :3]
target_points = rb.state[:e][:, 3:]
reached_points = rb.next_state[:e][:, :3]

target_error = np.linalg.norm(target_points - reached_points, axis=1)
target_distance = np.linalg.norm(target_points-com0_points, axis=1)

error = target_error/target_distance

if merged:
    fig = fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(projection='3d')
    ax.set_xlim(-.7, .7)
    ax.set_ylim(-.7, .7)
    ax.set_zlim(0, .6)

    ax.view_init(45, 45)

    ax.scatter(target_points[:, 0], target_points[:, 1], target_points[:, 2],
               color="blue", alpha=0.6, edgecolors="black", linewidths=0.2, label="target CoM test")
    ax.scatter(reached_points[:, 0], reached_points[:, 1], reached_points[:, 2],
               color="red", alpha=0.6, edgecolors="black", linewidths=0.2, label="reached CoM test")

    ax.legend()
    fig.savefig(os.path.join('plots', 'latest_merge.png'), dpi=300)
    plt.show()

else:
    if not filter:
        fig = fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(projection='3d')
        ax.set_xlim(-.7, .7)
        ax.set_ylim(-.7, .7)
        ax.set_zlim(0, .6)

        ax.view_init(45, 45)
        if target_e:
            test_distance = ax.scatter(target_points[:, 0], target_points[:, 1], target_points[:, 2],
                                       c=target_error, alpha=0.7, edgecolors="black", linewidths=0.2, cmap=cmap, norm=norm, label="reached CoM test")
        else:
            test_distance = ax.scatter(target_points[:, 0], target_points[:, 1], target_points[:, 2],
                                       c=error, alpha=0.7, edgecolors="black", linewidths=0.2, cmap=cmap, norm=norm, label="reached CoM test")
        cbar = plt.colorbar(test_distance, shrink=0.5)
        # cbar.set_label("Reward")
        ax.legend()
        fig.savefig(os.path.join('plots', 'latest_complete.png'), dpi=300)
        plt.show()

    elif filter:
        if target_e:
            idx = target_error <= err_treshold
            err_filter = target_error[idx]
        else:
            idx = error <= err_treshold
            err_filter = error[idx]
        feasible_region = target_points[idx, :]
        print(len(err_filter)/e)

        fig = fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(projection='3d')
        ax.set_xlim(-.65, .65)
        ax.set_ylim(-.65, .65)
        ax.set_zlim(0, .5)
        ax.view_init(45, 45)
        feasible_reward = ax.scatter(feasible_region[:, 0], feasible_region[:, 1], feasible_region[:, 2],
                                     c=err_filter, alpha=0.7, edgecolors="black", linewidths=0.2, cmap=cmap2, norm=norm2, label="targetCoM test")
        cbar = plt.colorbar(feasible_reward, shrink=0.5)
        # cbar.set_label("Reward")
        ax.legend()
        fig.savefig(os.path.join('plots', 'latest_filtered.png'), dpi=300)
        plt.show()
