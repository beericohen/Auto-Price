from DataCleaner import cleanData
from EDA import eda
from prepprocessing import prepprocessing
from train_and_tune import train_and_tune_neural_network
from export_model import main
cleanData()

eda()

prepprocessing()

train_and_tune_neural_network()

main()