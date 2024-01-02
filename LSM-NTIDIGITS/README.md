## Supervised N-TIDIGITS classification on LSM-based RRAM


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
| SRNN-ANN       | 82.62       |
| SRNN-SNN       | 63.75       |
| RNN            | 75.54       |
| **LSM (Software)** | **70.79**   |
| **LSM (RRAM)**     | **70.11**   |



