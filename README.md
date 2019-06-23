# gan-mnist
Developed my first unconditional Generative Adversarial Network to generate images similar to the mnist dataset.

## References:
Followed tutorial from 'Generative Adversarial Networks Cookbook' by Josh Kalin (via Packt Publishing)

## Dependencies:
python3
tensorflow
keras
graphviz
I used cuda9.0-devel since it matches the driver for my GTX 1050 mobile driver on Ubuntu 18.04 LTS
etc.

## To fix:
minor printing error with the image saving - i believe it should output 16x16 grids at each of the key epochs in /data/, but this doesn't seemto be the case - it seems to instead, overwrite over the same spot. minor fix required to the loop.
    fix location: FILE: train.py FUNCTION: plot_checkpoint(self,e)
