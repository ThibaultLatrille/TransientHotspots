python3 scripts/model.py --plot True --determinist True --pop_size 10000 --sel_coeff_shape -1 --b_cold 6e-6 --dominance_coeff 0.5 --nb_sim 1e3 --output figures/determinist_h0.5.tsv.gz --b_hot 0.01 --hotspot_lifespan 2250 --sel_coeff_mean 0.01
python3 scripts/model.py --plot True --pop_size 10000 --sel_coeff_shape -1 --b_cold 6e-6 --dominance_coeff 0.5 --nb_sim 4e3 --output figures/stochastic_h0.5.tsv.gz --b_hot 0.01 --hotspot_lifespan 2250 --sel_coeff_mean 0.01
python3 scripts/model.py --plot True --determinist True --pop_size 10000 --sel_coeff_shape -1 --b_cold 6e-6 --dominance_coeff 0.3 --nb_sim 1e3 --output figures/determinist_h0.3.tsv.gz --b_hot 0.01 --hotspot_lifespan 2250 --sel_coeff_mean 0.01
python3 scripts/model.py --plot True --pop_size 10000 --sel_coeff_shape -1 --b_cold 6e-6 --dominance_coeff 0.3 --nb_sim 4e3 --output figures/stochastic_h0.3.tsv.gz --b_hot 0.01 --hotspot_lifespan 2250 --sel_coeff_mean 0.01
