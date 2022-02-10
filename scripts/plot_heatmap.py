# GLOBAL IMPORTS
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_matrix(s_mesh, b_mesh, matrix, ylabel, filepath, vmax=None, vmin=None):
    # plt.figure(figsize=(1080 / 256, 1080 / 256), dpi=256)
    if vmax == vmin:
        plt.contourf(s_mesh, b_mesh, matrix, 20, cmap='RdGy')
    else:
        plt.contourf(s_mesh, b_mesh, matrix, 20, cmap='RdGy', vmax=vmax, vmin=vmin)

    plt.gca().yaxis.set_ticks(np.linspace(np.min(b_mesh), np.max(b_mesh), 10))
    plt.gca().xaxis.set_ticks(np.linspace(np.min(s_mesh), np.max(s_mesh), 6))
    plt.xlabel("s", fontsize=18)
    plt.ylabel("b", fontsize=18)
    cbar = plt.colorbar()
    cbar.set_label(ylabel, fontsize=18)
    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--input', required=True, type=str, dest="input")
    parser.add_argument('--output', required=True, type=str, dest="output")
    parser.add_argument('--load', required=False, type=bool, default=False, dest="load")
    args_parse = parser.parse_args()

    df = pd.read_csv(args_parse.input, sep="\t")
    if args_parse.load:
        df["pfix"] = df["pfix"] * df["sel_coeff"]
    m_grouped = {k: t.pivot(index="b_hot", columns="sel_coeff", values="pfix") for k, t in df.groupby(["lifespan"])}
    tau_max = max(m_grouped.keys())
    m_inf = m_grouped[tau_max]
    S, B = np.meshgrid(m_inf.index, m_inf.columns)
    scale = 1e4 if args_parse.load else 1
    t_infty = "L_{\\infty}" if args_parse.load else "x_{eq}"
    t_tau = "L_{\\tau}" if args_parse.load else "x_{\\tau}"
    plot_matrix(S, B, m_inf * scale, f"${t_infty}$", f"{args_parse.output}_inf.pdf")

    vmin = np.min([np.min((m - m_inf) * scale) for tau, m in m_grouped.items() if tau != tau_max])
    vmax = np.max([np.max((m - m_inf) * scale) for tau, m in m_grouped.items() if tau != tau_max])
    for tau, m in m_grouped.items():
        if tau == tau_max:
            continue
        plot_matrix(S, B, m, f"${t_tau}$", f"{args_parse.output}_{tau}.pdf")
        plot_matrix(S, B, (m - m_inf) * scale, f"${t_tau} - {t_infty}$ ", f"{args_parse.output}_{tau}-inf.pdf",
                    vmax=vmax, vmin=vmin)
