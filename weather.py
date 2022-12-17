# Will be using National Weather Service's API
# https://weather-gov.github.io/api/general-faqs

# For Geocoding will use Google's GeoCoding API

import keys

import googlemaps
import requests

NCS_FORECAST_ENDPOINT = "https://api.weather.gov/points/{lat},{lng}"

gmaps = googlemaps.Client(key=keys.google_map_key)

def get_location_coords(location):
    full_res = gmaps.geocode(location)
    return full_res[0]["geometry"]["location"]

def get_forecast_endpoints(lat, lng):
    full_res = requests.get(NCS_FORECAST_ENDPOINT.format(lat=lat, lng=lng)).json()
    endpoints = {
        "normal": full_res["properties"]["forecast"],
        "hourly": full_res["properties"]["forecastHourly"]
    }
    return endpoints

def get_weather(location):
    coords = get_location_coords(location)
    endpoints = get_forecast_endpoints(coords["lat"], coords["lng"])

    normal_weather = requests.get(endpoints["normal"]).json()

    return normal_weather["properties"]["periods"][0]["detailedForecast"]
