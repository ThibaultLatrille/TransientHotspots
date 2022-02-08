# GLOBAL IMPORTS
from collections import defaultdict
import argparse
import pandas as pd
import numpy as np
import time


def simulation(args):
    theta = args.sel_coeff_mean / args.sel_coeff_shape
    s = np.random.gamma(args.sel_coeff_shape, theta)
    der = 1
    t, t_max = 0, 100 * args.pop_size
    while der != 0 and der != 2 * args.pop_size:
        x = der / (2 * args.pop_size)
        b = args.b_hot if t < args.hotspot_lifespan else args.b_cold
        x_prime = (1 - s) * x ** 2 + (1 + b) * (1 - args.dominance_coeff * s) * x * (1 - x)
        der = np.random.binomial(2 * args.pop_size, min(1, x_prime))
        t += 1
        if t > t_max:
            return x, np.nan

    return der / (2 * args.pop_size), t


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--pop_size', required=False, type=int, default=10000, dest="pop_size")
    parser.add_argument('--sel_coeff_mean', required=False, type=float, default=0.0, dest="sel_coeff_mean")
    parser.add_argument('--sel_coeff_shape', required=False, type=float, default=0.0, dest="sel_coeff_shape")
    parser.add_argument('--b_hot', required=False, type=float, default=0.01, dest="b_hot")
    parser.add_argument('--b_cold', required=False, type=float, default=0.0001, dest="b_cold")
    parser.add_argument('--dominance_coeff', required=False, type=float, default=0.5, dest="dominance_coeff")
    parser.add_argument('--hotspot_lifespan', required=False, type=float, default=0.0, dest="hotspot_lifespan")
    parser.add_argument('--output', required=True, type=str, dest="output")
    args_parse = parser.parse_args()

    assert args_parse.sel_coeff_mean >= 0.0
    assert args_parse.sel_coeff_shape >= 0.0
    assert args_parse.dominance_coeff * args_parse.sel_coeff_mean < 1.0

    t_1 = time.perf_counter()
    nb_sim = 10 ** 6
    results = defaultdict(list)
    for replicate in range(nb_sim):
        x_f, t_f = simulation(args_parse)
        results["x_T"].append(x_f)
        results["T_fix"].append(t_f)
    pd.DataFrame(results).to_csv(args_parse.output, sep="\t")
    print(f"τ={args_parse.hotspot_lifespan}; Pfix={np.mean(results['x_T'])}.")
    print(f"{time.perf_counter() - t_1:.2f}s for {nb_sim} simulations.")
