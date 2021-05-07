# LeNet
The original LeNet specification

## What is saved
- train and val loss and accuracy per epoch
- weights of best epochs
- log
TODO

## Usage
### training
python main.py {EPOCHS} {MODEL NAME} > log.txt; source movelog.sh {MODEL NAME}

### evaluating
python main.py -1 {MODEL NAME} {SAVED WEIGHTS NAME}


## Saved Weights
- test1: temporary file for testing code correctness, please ignore