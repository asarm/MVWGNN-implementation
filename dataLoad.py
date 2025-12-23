import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr, circmean, circstd
from sklearn.impute import KNNImputer

def load_data(dataName="hourly-data"):
        # load necessary data separately
        if dataName == "hourly-data":
            startDate = '2012-10-02'
            endDate = '2017-10-28'
            path = "data/"
        elif dataName == "ceda_data":
            startDate = '2019-01-01'
            endDate = '2024-12-31'
            path = "ceda_data/"

        cities = pd.read_csv(f"{path}city_attributes.csv")
        cities = cities[cities.Country != 'Israel']
        city_names = cities['City'].tolist()
        
        # columns are datetime + city names
        humidity = pd.read_csv(f"{path}humidity.csv")
        humidity['datetime'] = pd.to_datetime(humidity['datetime'])
        humidity = humidity[humidity['datetime'] >= startDate]
        humidity = humidity[humidity['datetime'] <= endDate]

        pressure = pd.read_csv(f"{path}pressure.csv")
        pressure['datetime'] = pd.to_datetime(pressure['datetime'])
        pressure = pressure[pressure['datetime'] >= startDate]
        pressure = pressure[pressure['datetime'] <= endDate]

        temperature = pd.read_csv(f"{path}temperature.csv")
        temperature['datetime'] = pd.to_datetime(temperature['datetime'])
        temperature = temperature[temperature['datetime'] >= startDate]
        temperature = temperature[temperature['datetime'] <= endDate]

        wind_direction = pd.read_csv(f"{path}wind_direction.csv")
        wind_direction['datetime'] = pd.to_datetime(wind_direction['datetime'])
        wind_direction = wind_direction[wind_direction['datetime'] >= startDate]
        wind_direction = wind_direction[wind_direction['datetime'] <= endDate]

        wind_speed = pd.read_csv(f"{path}wind_speed.csv")
        wind_speed['datetime'] = pd.to_datetime(wind_speed['datetime'])
        wind_speed = wind_speed[wind_speed['datetime'] >= startDate]
        wind_speed = wind_speed[wind_speed['datetime'] <= endDate]

        features = {
            'humidity': humidity,
            'pressure': pressure,
            'temperature': temperature,
            'wind_direction': wind_direction,
            'wind_speed': wind_speed
        }

        # Impute missing values
        '''
        for feature_name, df in features.items():
            data_array = df[city_names].values  # shape (T, n_stations)
            imputer = KNNImputer(n_neighbors=5)
            imputed_array = imputer.fit_transform(data_array)
            features[feature_name][city_names] = imputed_array
        '''
        
        # Prepare a dictionary for MultiIndex columns
        data = {}
        for city in city_names:
            for feature, df in features.items():
                data[(city, feature)] = df[city].interpolate(method='linear', limit_direction='both')

        df = pd.DataFrame(data)
        df.columns = pd.MultiIndex.from_tuples(df.columns)

        df.index = humidity['datetime']
        df.index.name = 'datetime'

        return df, cities