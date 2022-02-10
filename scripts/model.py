# GLOBAL IMPORTS
from collections import defaultdict
import argparse
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt


def x_prime(x, s, b, h):
    xp = (1 - s) * x ** 2 + (1 + b) * (1 - h * s) * x * (1 - x)
    xp /= (1 - s * x * x - 2 * h * s * x * (1.0 - x))
    return min(1.0, max(0.0, xp))


def simulation_det(args, replicate):
    x, x_list = 1 / (2 * args.pop_size), []
    for t in range(args.hotspot_lifespan * (2 if args.plot else 1)):
        b = args.b_hot if t < args.hotspot_lifespan else args.b_cold
        if args.plot:
            x_list.append(x)
            if replicate == 0:
                b = args.b_hot
        x = x_prime(x, args.sel_coeff_mean, b, args.dominance_coeff)
    if args.plot:
        plt.plot(range(len(x_list)), x_list, ls=('--' if replicate == 0 else "-"),
                 color=('tab:blue' if replicate == 0 else "tab:orange"))
    return x, args.hotspot_lifespan


def simulation_stoch(args):
    s = args.sel_coeff_mean
    if args.sel_coeff_shape > 0:
        theta = args.sel_coeff_mean / args.sel_coeff_shape
        while True:
            s = np.random.gamma(args.sel_coeff_shape, theta)
            if s < 1:
                break
    der, x_list = 1, []
    t, t_max = 0, 100 * args.pop_size
    while der != 0 and der != 2 * args.pop_size:
        x = der / (2 * args.pop_size)
        if args.plot:
            x_list.append(x)
        b = args.b_hot if t < args.hotspot_lifespan else args.b_cold
        der = np.random.binomial(2 * args.pop_size, x_prime(x, s, b, args.dominance_coeff))
        t += 1
        if t > t_max:
            return x, np.nan
    if args.plot:
        plt.plot(range(len(x_list)), x_list, lw=0.3,
                 color=('tab:blue' if der == 2 * args.pop_size else "tab:orange"),
                 ls=('--' if der == 2 * args.pop_size else "-"))
    return der / (2 * args.pop_size), t


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--pop_size', required=False, type=int, default=10000, dest="pop_size")
    parser.add_argument('--nb_sim', required=False, type=float, default=10000, dest="nb_sim")
    parser.add_argument('--sel_coeff_mean', required=False, type=float, default=0.0, dest="sel_coeff_mean")
    parser.add_argument('--sel_coeff_shape', required=False, type=float, default=-1.0, dest="sel_coeff_shape")
    parser.add_argument('--b_hot', required=False, type=float, default=0.01, dest="b_hot")
    parser.add_argument('--b_cold', required=False, type=float, default=0.0001, dest="b_cold")
    parser.add_argument('--dominance_coeff', required=False, type=float, default=0.5, dest="dominance_coeff")
    parser.add_argument('--hotspot_lifespan', required=False, type=int, default=0, dest="hotspot_lifespan")
    parser.add_argument('--determinist', required=False, type=bool, default=False, dest="determinist")
    parser.add_argument('--plot', required=False, type=bool, default=False, dest="plot")
    parser.add_argument('--output', required=True, type=str, dest="output")
    args_parse = parser.parse_args()

    assert args_parse.sel_coeff_mean >= 0.0
    assert args_parse.dominance_coeff * args_parse.sel_coeff_mean < 1.0
    if args_parse.determinist:
        args_parse.nb_sim = 2
    if args_parse.plot:
        my_dpi = 256 + 128
        plt.figure(figsize=(1920 / my_dpi, 1080 / my_dpi), dpi=my_dpi)

    t_1 = time.perf_counter()
    results = defaultdict(list)
    for i in range(int(args_parse.nb_sim)):
        x_f, t_f = simulation_det(args_parse, i) if args_parse.determinist else simulation_stoch(args_parse)
        results["x_T"].append(x_f)
        results["T_fix"].append(t_f)
    pd.DataFrame(results).to_csv(args_parse.output, sep="\t", compression="gzip")
    if args_parse.plot:
        plt.axvline(args_parse.hotspot_lifespan, color="black")
        plt.xlabel("$t$")
        plt.ylabel("$x_t$")
        plt.ylim((0, 1))
        plt.tight_layout()
        plt.title(f"s={args_parse.sel_coeff_mean}")
        plt.savefig(args_parse.output.replace(".tsv.gz", ".pdf"))
    print(f"τ={args_parse.hotspot_lifespan}; Pfix={np.mean(results['x_T'])}.")
    print(f"{time.perf_counter() - t_1:.2f}s for {args_parse.nb_sim} simulations.")
