# ciwganfiwgan-pytorch

This is a PyTorch implementation of **Categorical Info WaveGAN (ciwGAN)** and **Featural Info WaveGAN (fiwGAN)** from [Beguš, 2021](https://www.sciencedirect.com/science/article/pii/S0893608021001052). The original code (in Tensorflow 1.12) can be found [here](https://github.com/gbegus/fiwGAN-ciwGAN). In this fork I invetigate the use of a 2nd Q network 

## Usage

### Training ciwGAN with an external Q-network for supervision

```
python -m pdb -c c train_Q2.py --ciw --Q2 --num_categ 11 --datadir  ~/notebooks/talker_variability/TIMIT_padded/ --logdir logs_q2
```


Add `--cont last` to the end of the training statement to continue from the last found state  
Add `--cont epoch_number` to continue from the state corresponding to `epoch_number`
