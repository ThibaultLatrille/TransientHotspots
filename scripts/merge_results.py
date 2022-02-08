# GLOBAL IMPORTS
from collections import defaultdict
import argparse
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-i', '--input', required=True, type=str, nargs="+", dest="input")
    parser.add_argument('-o', '--output', required=True, type=str, dest="output")
    args_parse = parser.parse_args()

    output_dict = defaultdict(list)
    for file in args_parse.input:
        df = pd.read_csv(file, sep="\t")
        lifespan = float(os.path.basename(file).replace(".tsv", ""))
        pfix = np.mean(df["x_T"])
        t_fix = np.nanmean(df["T_fix"])
        output_dict["pfix"].append(pfix)
        output_dict["lifespan"].append(lifespan)
        output_dict["T_fix"].append(t_fix)

    df_out = pd.DataFrame(output_dict)
    df_out.to_csv(args_parse.output)

    plt.scatter(df_out["lifespan"], df_out["pfix"])
    plt.xlabel("$\\tau$")
    plt.ylabel("$P_{fix}$")
    plt.savefig(args_parse.output.replace(".tsv", ".p_fix.pdf"))
    plt.close()

    plt.scatter(df_out["lifespan"], df_out["T_fix"])
    plt.xlabel("$\\tau$")
    plt.ylabel("$T_{absorption}$")
    plt.savefig(args_parse.output.replace(".tsv", ".t_absorption.pdf"))
