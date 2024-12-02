from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import joblib
import time
import requests as req
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


# Define class for interacting with the Meteo API
class WeatherData:
    """
    A class to fetch and process meteorological data from the https://api.meteo.lt/ API.
    """

    def __init__(self, historical_url, forecast_url):
        self.historical_url = historical_url
        self.forecast_url = forecast_url

    def __fetch_data(self, url):
        try:
            response = req.get(url=url, timeout=1000)
        except Exception as error:
            print(f"Error fetching data: {error}")
        else:
            if response.status_code != 200:
                print(f"Failed to fetch data. HTTP status: {response.status_code}")
            else:
                return response.json()

    def load_historical_data(self, start_date, end_date):
        start_date = datetime.strptime(start_date, "%Y-%m-%d")
        end_date = datetime.strptime(end_date, "%Y-%m-%d")
        current_date = start_date
        weather_records = []
        while current_date < end_date:
            url = f'{self.historical_url}/{current_date.strftime("%Y-%m-%d")}'
            weather_records += self.__fetch_data(url)["observations"]
            current_date += timedelta(days=1)
            time.sleep(0.5)  # Pause to respect API rate limits
        weather_df = pd.DataFrame(weather_records)
        weather_df = weather_df.set_index("observationTimeUtc", drop=True)
        weather_df.index = (
            pd.to_datetime(weather_df.index)
            .tz_localize("UTC")
            .tz_convert("Europe/Vilnius")
            .tz_localize(None)
        )
        weather_df.index.name = None
        return weather_df

    def load_forecast_data(self):
        forecast_data = self.__fetch_data(self.forecast_url)
        forecast_df = pd.DataFrame(forecast_data["forecastTimestamps"])
        forecast_df = forecast_df.set_index("forecastTimeUtc", drop=True)
        forecast_df.index = (
            pd.to_datetime(forecast_df.index)
            .tz_localize("UTC")
            .tz_convert("Europe/Vilnius")
            .tz_localize(None)
        )
        forecast_df.index.name = None
        return forecast_df


# Main program logic
if __name__ == "__main__":
    HISTORICAL_API_URL = "https://api.meteo.lt/v1/stations/vilniaus-ams/observations"
    FORECAST_API_URL = "https://api.meteo.lt/v1/places/vilnius/forecasts/long-term"
    weather_api = WeatherData(HISTORICAL_API_URL, FORECAST_API_URL)

    # Define the date range for historical data
    today = datetime.now() + timedelta(days=1)
    one_year_ago = today - relativedelta(years=1)
    today_str = today.strftime("%Y-%m-%d")
    one_year_ago_str = one_year_ago.strftime("%Y-%m-%d")

    # Fetch and save data
    historical_weather = weather_api.load_historical_data(one_year_ago_str, today_str)
    forecast_weather = weather_api.load_forecast_data()

    joblib.dump(historical_weather, "historical_weather.pkl")
    joblib.dump(forecast_weather, "forecast_weather.pkl")

    # Calculate average values
    print(historical_weather[["airTemperature", "relativeHumidity"]].mean())
    day_start = pd.to_datetime("08:00:00").time() <= historical_weather.index.time
    day_end = pd.to_datetime("20:00:00").time() > historical_weather.index.time
    print(
        "Daytime average temperature:",
        historical_weather["airTemperature"][day_start & day_end].mean(),
    )
    print(
        "Nighttime average temperature:",
        historical_weather["airTemperature"][~day_end | ~day_start].mean(),
    )

    # Identify rainy weekends
    saturday_weather = historical_weather["conditionCode"][
        historical_weather.index.weekday == 5
    ]
    sunday_weather = historical_weather["conditionCode"][
        historical_weather.index.weekday == 6
    ]
    sunday_weather.index -= timedelta(days=1)  # Align Sundays with Saturdays
    combined_weekend = pd.merge(
        saturday_weather, sunday_weather, left_index=True, right_index=True, how="outer"
    )
    rainy_sat = combined_weekend["conditionCode_x"].apply(lambda x: "rain" in str(x))
    rainy_sun = combined_weekend["conditionCode_y"].apply(lambda x: "rain" in str(x))
    rainy_combined = (rainy_sat | rainy_sun).to_frame()
    rainy_combined["date"] = rainy_combined.index.date
    rainy_weekend_count = (
        rainy_combined.groupby("date")
        .agg(list)
        .unstack()
        .apply(lambda x: True in x)
        .sum()
    )
    print(f"Number of rainy weekends: {rainy_weekend_count}")

    # Combine historical and forecast data
    all_weather = pd.concat([historical_weather, forecast_weather], axis=0)
    all_weather = all_weather.loc[~all_weather.index.duplicated(keep="first")]
    last_week = datetime.now() - timedelta(days=7)
    next_week = datetime.now() + timedelta(days=7)

    # Plot the temperature trend
    plt.figure(figsize=(12, 6))
    plt.plot(all_weather[last_week:next_week]["airTemperature"])
    plt.title("Temperature Trend")
    plt.xlabel("Date")
    plt.ylabel("Temperature (°C)")
    plt.xticks(rotation=45)
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=1))
    plt.show()

    # Custom interpolation functions
    def custom_linear_interpolation(series, interpolation_type):
        resampled_index = series.resample("5min").asfreq().index
        if interpolation_type == "linear":
            interpolated_data = np.array(series.values[0])[np.newaxis]
            for s, e in zip(series.values[:-1], series.values[1:]):
                interpolated_segment = (e - s) / 12 * np.arange(1, 12, 1) + s
                interpolated_data = np.concatenate(
                    [interpolated_data, interpolated_segment, e[np.newaxis]], axis=0
                )
            return pd.Series(interpolated_data, index=resampled_index)

    def built_in_interpolation(series, interpolation_method):
        resampled_series = (
            series.resample("5min").asfreq().interpolate(method=interpolation_method)
        )
        return resampled_series

    linear_interp = custom_linear_interpolation(
        forecast_weather[:10]["airTemperature"], "linear"
    )
    print(linear_interp[:20])
    pandas_interp = built_in_interpolation(
        forecast_weather[:10]["airTemperature"], "linear"
    )
    print(pandas_interp[:20])
