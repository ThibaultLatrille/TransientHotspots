# GLOBAL IMPORTS
import os
import argparse
import numpy as np
import pandas as pd
from collections import defaultdict

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--input', required=True, type=str, dest="input")
    parser.add_argument('--output', required=True, type=str, dest="output")
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
