import pandas as pd
import numpy as np
from fbprophet import Prophet

raw_data = pd.read_csv('data/Tainan_weather_station.csv')
raw_data.columns = ['stn_no', 'date', 'temp', 'wind_spd', 'wind_dir', 'sun_rate', 'solar_irr']

data = raw_data.copy()

for col in data.columns[2:]:
    mean = data[col].mean()
    data[col][data[col] < 0] = mean

data = data.groupby(['date'])[data.columns[2:]].mean()

df = data['wind_spd'].to_frame()
df.reset_index(level='date', inplace=True)

df.rename(columns={'date':"ds", 'wind_spd':'y'}, inplace=True)
df['ds'] = pd.to_datetime(df['ds'], format='%Y%m%d')

# data preporc
# shift = 3.0164
# df['y'] = df['y'] - shift
# df['y'] = np.log(df['y'])
# df['y'] = (df['y'] - df['y'].mean()) / df['y'].std()

# prophet analysis
m = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
m.fit(df)

future = m.make_future_dataframe(periods=365, freq='D')
future.tail()
forecast = m.predict(future)

plt = m.plot(forecast)
plt.savefig('figs/forecast.png')

plt = m.plot_components(forecast)
plt.savefig('figs/component.png')
