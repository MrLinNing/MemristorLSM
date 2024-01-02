## Supervised N-MNIST classification on LSM-based RRAM


### Baseline models consist of SRNN-ANN, SRNN-SNN models and RNN
```angular2html
bash run_srnn_ann.sh
bash run_srnn_snn.sh
bash run_rnn.sh
```

### LSM simulation on software with Gaussian Distribution
```angular2html
bash run_lsm_gau_sim.sh
```

### LSM run RRAM
```angular2html
bash run_lsm_rram.sh
```

### Accuracy Comparisions

| Model          | Accuracy(%) |
|----------------|-------------|
| SRNN-ANN       | 96.96       |
| SRNN-SNN       | 93.93       |
| RNN            | 96.23       |
| **LSM (Software)** | **89.11**       |
| **LSM (RRAM)**     | **89.66**       |


