# GraphAT
Graph Adversarial Training: Dynamically Regularizing Based on Graph Structure


Python 3.6.1 :: Continuum Analytics, Inc.
tensorflow                         1.8.0
numpy                              1.18.1

GCN-VAT

Cora

python vat_citation.py --epsilon 0.01 --alpha 1.0 --xi 0.001 --dropout 0.0 --dataset cora --early_stopping 10

Citeseer
python vat_citation.py --epsilon 0.05 --alpha 0.5 --xi 0.0001 --dropout 0.0 --dataset citeseer --early_stopping 10

GraphAT

Cora

python gvat_citation.py --gat_loss=True --num_neighbors 2 --epsilon_graph 0.01 --beta 1.0 --dropout 0.0 --dataset cora --early_stopping 10

Citeseer

python gvat_citation.py --gat_loss=True --num_neighbors 2 --epsilon_graph 0.01 --beta 0.5 --dropout 0.0 --dataset citeseer --early_stopping 10

GraphVAT

Cora

python gvat_citation.py --gat_loss=True --vat_loss=True --epsilon 1.0 --alpha 0.5 --xi 1e-05 --num_neighbors 2 --epsilon_graph 0.01 --beta 1.0 --dropout 0.0 --dataset cora --early_stopping 10

Citeseer
python gvat_citation.py --gat_loss=True --vat_loss=True --epsilon 1.0 --alpha 0.5 --xi 1e-06 --num_neighbors 2 --epsilon_graph 0.01 --beta 0.5 --dropout 0.0 --dataset citeseer --early_stopping 1
