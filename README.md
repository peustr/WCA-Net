# WCA-Net

## How to train a new model

```

python run.py train <path to configuration file>

```

## Configuration files

- m0: Vanilla model (backbone + classification layer)
- m1: Stochastic isotropic model, trained with WCA
- m2: Stochastic anisotropic model, trained with WCA
- m3: Stochastic anisotropic model, trained with WCA, adversarial training
