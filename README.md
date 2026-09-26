# pytorch-experiments
experiments with pytorch and other libraries

## notebooks

| notebook | topic |
|---|---|
| `pytorch_basics.ipynb` | tensor fundamentals (scalar → vector → matrix → tensor), ops, creation, item extraction, exercises; and a first regression workflow: linear model, L1Loss + SGD, train/infer/save |
| `pytorch_nonlinear.ipynb` | stacked `nn.Linear` layers + ReLU instead of a single weight/bias; fits a curve the old model can't; SGD vs Adam comparison |
| `pytorch_classification.ipynb` | binary classification on `make_circles`: logits + sigmoid, `BCEWithLogitsLoss`, accuracy metric, why a linear-only model fails (~38% vs ~95%) |

built on the [learnpytorch.io](https://www.learnpytorch.io/) curriculum (chapters 00-02 so far); next up: multi-class classification, then computer vision with `Dataset`/`DataLoader`.

requirements: `torch`, `matplotlib`, `scikit-learn` (`numpy` comes along automatically). notebooks run on cpu or gpu (they use `device = "cuda" if torch.cuda.is_available() else "cpu"`).
