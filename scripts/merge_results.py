# GLOBAL IMPORTS
from collections import defaultdict
import argparse
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--pop_size', required=False, type=int, default=10000, dest="pop_size")
    parser.add_argument('-i', '--input', required=True, type=str, dest="input")
    parser.add_argument('-o', '--output', required=True, type=str, dest="output")
    args_parse = parser.parse_args()

    output_dict = defaultdict(list)
    for file in os.listdir(args_parse.input):
        if not file.endswith("tsv.gz"):
            continue
        df_read = pd.read_csv(f"{args_parse.input}/{file}", sep="\t", compression="gzip")
        b_hot, sel_coeff, lifespan = os.path.basename(file).replace(".tsv.gz", "").split("_")
        pfix = np.mean(df_read["x_T"])
        t_fix = np.nanmean(df_read["T_fix"])
        output_dict["b_hot"].append(float(b_hot))
        output_dict["sel_coeff"].append(float(sel_coeff))
        output_dict["lifespan"].append(int(lifespan))
        output_dict["pfix"].append(pfix)
        output_dict["T_fix"].append(t_fix)

    df_out = pd.DataFrame(output_dict)
    df_out.to_csv(args_parse.output, sep="\t")

    df_grouped = {k: table for k, table in df_out.groupby(["lifespan"])}
    df_inf = df_grouped[max(df_grouped.keys())]
    matrix_inf = df_inf.pivot(index="b_hot", columns="sel_coeff", values="pfix")
    for lifespan, df in df_grouped.items():
        if lifespan == max(df_grouped.keys()):
            continue
        matrix = df.pivot(index="b_hot", columns="sel_coeff", values="pfix")
        S, B = np.meshgrid(matrix.index, matrix.columns)
        for g in [0.0, 0.2, 0.8]:
            data = gaussian_filter(matrix - matrix_inf, g)
            plt.contourf(S, B, data, 20, cmap='RdGy')
            plt.xlabel("s")
            plt.ylabel("b")
            cbar = plt.colorbar()
            cbar.set_label("$x_{\\tau} - x_{eq}$")
            plt.tight_layout()
            plt.savefig(args_parse.output.replace(".tsv", f"{lifespan}.{g}.pdf"))
            plt.close()
