# GraphAT

Code for the GraphAT, GraphVAT, and GCN-VAT in our paper "Graph Adversarial Training: Dynamically Regularizing Based on Graph Structure", [\[paper\]](https://arxiv.org/abs/1902.08226).

## Environment

Python 3.6.1 :: Continuum Analytics, Inc.

tensorflow                         1.8.0

numpy                              1.18.1

## Reproduction
Once configured the required environment, the prediction performance reported in our paper can be reproduced by running the following commands (Table 4).

### GraphAT

Cora
```
python gvat_citation.py --gat_loss=True --num_neighbors 2 --epsilon_graph 0.01 --beta 1.0 --dropout 0.0 --dataset cora --early_stopping 10
```

Citeseer
```
python gvat_citation.py --gat_loss=True --num_neighbors 2 --epsilon_graph 0.01 --beta 0.5 --dropout 0.0 --dataset citeseer --early_stopping 10
```

### GraphVAT

Cora
```
python gvat_citation.py --gat_loss=True --vat_loss=True --epsilon 1.0 --alpha 0.5 --xi 1e-05 --num_neighbors 2 --epsilon_graph 0.01 --beta 1.0 --dropout 0.0 --dataset cora --early_stopping 10
```

Citeseer
```
python gvat_citation.py --gat_loss=True --vat_loss=True --epsilon 1.0 --alpha 0.5 --xi 1e-06 --num_neighbors 2 --epsilon_graph 0.01 --beta 0.5 --dropout 0.0 --dataset citeseer --early_stopping 1
```

### GCN-VAT

Cora
```
python vat_citation.py --epsilon 0.01 --alpha 1.0 --xi 0.001 --dropout 0.0 --dataset cora --early_stopping 10
```

Citeseer
```
python vat_citation.py --epsilon 0.05 --alpha 0.5 --xi 0.0001 --dropout 0.0 --dataset citeseer --early_stopping 10
```

## Cite

If you use the code, please kindly cite the following paper:
```
@article{feng2019graph,
  title={Graph adversarial training: Dynamically regularizing based on graph structure},
  author={Feng, Fuli and He, Xiangnan and Tang, Jie and Chua, Tat-Seng},
  journal={IEEE Transactions on Knowledge and Data Engineering},
  year={2019},
  publisher={IEEE}
}
```

## Contact

fulifeng93@gmail.com
