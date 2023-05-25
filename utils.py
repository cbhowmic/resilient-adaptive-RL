import numpy as np
import random
from shapely.geometry import Point

def random_point_set(n, lower=-10, upper=10):
    points = []
    assert lower <= upper
    for i in range(n):
        x = random.uniform(lower, upper)
        y = random.uniform(lower, upper)
        points.append(Point(x, y))
    return points


def findNeighbors(x, k, numAgents, rmax, maxNeighborSize=10):
    N = [k]
    for i in range(numAgents):
        if i == k:
            continue
        n = x[i]
        if np.sqrt((n.y - x[k].y) ** 2 + (n.x - x[k].x) ** 2) <= rmax:
            N.append(i)

    if len(N) > maxNeighborSize:
        selection = random.sample(N[1:], maxNeighborSize - 1)
        return [N[0]] + selection
    else:
        return N


def hard_target_update(main, target):
    target.load_state_dict(main.state_dict())


def soft_update(target, source, tau=0.005):
    """
    Copies the parameters from source network (x) to target network (y) using the below update
    y = TAU*x + (1 - TAU)*y
    """
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - tau) + param.data * tau
        )
