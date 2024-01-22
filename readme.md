Paper [link](https://www.sciencedirect.com/science/article/pii/S2352914823000199)


# To run the code execute:
Install required packages with:
```
!pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu
!pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.12.0+cpu.html
```

Execute the model with required parameters. The parameters are expalined in both paper and code.

```
!python main.py --epochs 20 --same False --negative-split fold  --l 1 --dataset LAGCN  lr 0.001 --thr-percent 2.5 --n 168  --agg-hetero sum --agg-conv var --agg-lin mul
```


