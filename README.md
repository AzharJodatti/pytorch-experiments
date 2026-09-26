# pytorch-experiments
experiments with pytorch and other libraries

## notebooks

| notebook | topic |
|---|---|
| `pytorch_basics.ipynb` | tensor fundamentals (scalar → vector → matrix → tensor), ops, creation, item extraction, exercises; and a first regression workflow: linear model, L1Loss + SGD, train/infer/save |
| `pytorch_nonlinear.ipynb` | stacked `nn.Linear` layers + ReLU instead of a single weight/bias; fits a curve the old model can't; SGD vs Adam comparison |
| `pytorch_classification.ipynb` | binary classification on `make_circles`: logits + sigmoid, `BCEWithLogitsLoss`, accuracy metric, why a linear-only model fails (~38% vs ~95%) |
| `pytorch_multiclass_classification.ipynb` | multi-class (4 blobs + hand-rolled three spirals): `nn.CrossEntropyLoss` on raw logits + class indices, softmax/argmax; blobs 100%, spirals 82.8% |
| `pytorch_computer_vision.ipynb` | FashionMNIST with `Dataset`/`DataLoader`, batching; linear baseline 83.9% vs small CNN 85.0%, confusion matrix, prediction gallery |
| `pytorch_learning_rates.ipynb` | Adam lr sweep 1e-6…1.0 on a curve + sine: too slow / just right / too hot loss curves and the V-shaped final-loss-vs-lr sweet-spot plot |
| `pytorch_gradient_descent_viz.ipynb` | what SGD actually does: MSE(w,b) loss surface in 3D + contour, hand-rolled autograd GD paths — crawl / steady / overshoot / divergence |
| `pytorch_device_reproducibility.ipynb` | plumbing: device `.to()` and the mismatch error, wall-clock vs `torch.cuda.Event` timing, `set_seeds()` (torch + numpy), determinism flags, dtypes and `element_size()` |

built on the [learnpytorch.io](https://www.learnpytorch.io/) curriculum (chapters 00-04 territory covered); next up: custom datasets, transfer learning, and experiment tracking.

requirements: `torch`, `matplotlib`, `scikit-learn` (`numpy` comes along automatically). notebooks run on cpu or gpu (they use `device = "cuda" if torch.cuda.is_available() else "cpu"`).
