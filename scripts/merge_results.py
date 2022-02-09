# GLOBAL IMPORTS
from collections import defaultdict
import argparse
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--pop_size', required=False, type=int, default=10000, dest="pop_size")
    parser.add_argument('-i', '--input', required=True, type=str, nargs="+", dest="input")
    parser.add_argument('-o', '--output', required=True, type=str, dest="output")
    args_parse = parser.parse_args()

    output_dict = defaultdict(list)
    for file in args_parse.input:
        df = pd.read_csv(file, sep="\t", compression="gzip")
        b_hot, sel_coeff, lifespan = os.path.basename(file).replace(".tsv.gz", "").split("_")
        pfix = np.mean(df["x_T"])
        t_fix = np.nanmean(df["T_fix"])
        output_dict["b_hot"].append(float(b_hot))
        output_dict["sel_coeff"].append(float(sel_coeff))
        output_dict["lifespan"].append(float(lifespan))
        output_dict["pfix"].append(pfix)
        output_dict["T_fix"].append(t_fix)

    df_out = pd.DataFrame(output_dict)
    df_out.to_csv(args_parse.output, sep="\t")

    plt.scatter(df_out["lifespan"], df_out["pfix"])
    plt.xlabel("$\\tau$")
    plt.ylabel("$P_{fix}$")
    assert len(set(df_out["b_hot"])) == len(set(df_out["sel_coeff"])) == 1
    b = set(df_out["b_hot"]).pop()
    s = set(df_out["sel_coeff"]).pop() * 0.5
    pfix_strong = 2 * (b - s) / (1.0 - np.exp(-4 * args_parse.pop_size * (b - s)))
    pfix_neutral = 1 / (2 * args_parse.pop_size)
    plt.axhline(pfix_strong, color="black", label="$P_{fix}^{strong}$")
    plt.axhline(pfix_neutral, color="grey", label="$P_{fix}^{neutral}$")
    plt.legend()
    plt.savefig(args_parse.output.replace(".tsv", ".p_fix.pdf"))
    plt.close()

    plt.scatter(df_out["lifespan"], df_out["T_fix"])
    plt.xlabel("$\\tau$")
    plt.ylabel("$T_{absorption}$")
    plt.savefig(args_parse.output.replace(".tsv", ".t_absorption.pdf"))