import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr, circmean, circstd

def load_data(dataName="hourly-data"):
    if dataName == "hourly-data":
        # load necessary data separately
        cities = pd.read_csv("hourly-data/city_attributes.csv")
        cities = cities[cities.Country != 'Israel']
        city_names = cities['City'].tolist()
        
        # columns are datetime + city names
        humity = pd.read_csv("hourly-data/humidity.csv")
        pressure = pd.read_csv("hourly-data/pressure.csv")
        temperature = pd.read_csv("hourly-data/temperature.csv")
        wind_direction = pd.read_csv("hourly-data/wind_direction.csv")
        wind_speed = pd.read_csv("hourly-data/wind_speed.csv")

        features = {
            'humidity': humity.ffill().bfill(),
            'pressure': pressure.ffill().bfill(),
            'temperature': temperature.ffill().bfill(),
            'wind_direction': wind_direction.ffill().bfill(),
            'wind_speed': wind_speed.ffill().bfill()
        }
        # city_names = [col for col in humity.columns if col != 'datetime']

        # Prepare a dictionary for MultiIndex columns
        data = {}
        for city in city_names:
            for feature, df in features.items():
                data[(city, feature)] = df[city]

        df = pd.DataFrame(data)
        df.columns = pd.MultiIndex.from_tuples(df.columns)

        df.index = humity['datetime']
        df.index.name = 'datetime'
    
    return df, cities