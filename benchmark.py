import pandas as pd
import matplotlib.pyplot as plt

from neuralforecast import NeuralForecast
from neuralforecast.models import LSTM, NBEATS, TFT, NLinear, TimesNet, FEDformer, FEDformer
# TODO: Add arima with seasonality handling