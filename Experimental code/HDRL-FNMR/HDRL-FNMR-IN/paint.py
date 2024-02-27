# -*- coding: utf-8 -*-
# @File    : paint.py
# @Date    : 2023-02-13
# @Author  : chenwei    -剑衣沉沉晚霞归，酒杖津津神仙来-
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import imageio


# plt.style.use('seaborn-whitegrid')


def paint_intrinsic_state(state, name, episode, index, step_counter):
    # state shape: 1, 5, 14, 14
    state = state.squeeze(0).numpy()  # 5, 14, 14
    channel1 = state[0] / 255  # subgoal
    channel2 = state[1] / 255  # route
    channel3 = state[2] + state[3] + state[4]  # metrics
    mask = np.where(channel3 != 0)

    channel3 = ((channel3 - channel3[mask].min()) /
                (channel3[mask].max() - channel3[mask].min()))
    channel3 -= channel3 * np.identity(14)
    mask = np.where(channel3 < 0)
    channel3[mask] = 0

    # RGB
    img = np.stack([channel2, channel1, channel3], axis=2)

    plt.xticks(range(14), range(1, 15))
    plt.yticks(range(14), range(1, 15))

    plt.title(f'episode:{episode} index:{index} step_counter:{step_counter}')
    plt.imshow(img, interpolation='nearest')
    Path(f'./images/{name}').mkdir(exist_ok=True, parents=True)
    plt.savefig(f'./images/{name}/{episode}_{index}_{step_counter}', )


def paint_gif(path='./images', match_str="*images/1*"):
    data_path = Path(path)

    imgs = []
    for img_path in data_path.iterdir():
        if img_path.match(match_str):
            print(img_path)
            img = imageio.imread(img_path)
            imgs.append(img)

    imageio.mimsave(data_path / 'test.gif', imgs, fps=30)


if __name__ == '__main__':
    paint_gif()
