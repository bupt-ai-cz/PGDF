# PGDF

## Training
Take CIFAR-10 with 50% symmetric noise as an example:

First, please modify the `data_path` in ``presets.json`` to indicate the location of your dataset.

Then, run
```bash
python train_cifar_getPrior.py --preset c10.50sym
```
to get the prior knowledge. Related files will be saved in ``checkpoints/c10/50sym/saved/``.

Next, run
```bash
python train_cifar.py --preset c10.50sym
```
for the subsequent training process.

``c10`` means CIFAR-10, ``50sym`` means 50% symmetric noise.  
Similarly, if you want to experiment on CIFAR-100 with 20% symmetric noise, you can use the command:
```bash
python train_cifar_getPrior.py --preset c100.20sym
```
```bash
python train_cifar.py --preset c100.20sym
```
## Additional Info
The (basic) semi-supervised learning part of our code is borrow from [the official DM-AugDesc implementation](https://github.com/KentoNishi/Augmentation-for-LNL/).

Since this paper is still being submitted, we only release part of the experimental code. We will release all the experimental codes after being accepted.