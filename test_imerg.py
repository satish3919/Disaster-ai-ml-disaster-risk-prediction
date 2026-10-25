import requests

def get_power_rainfall(lat, lon):
    url = "https://power.larc.nasa.gov/api/temporal/daily/point"
    params = {
        "parameters": "PRECTOTCORR",
        "community": "AG",
        "longitude": lon,
        "latitude": lat,
        "start": "20260910",
        "end": "20260917",
        "format": "JSON"
    }
    response = requests.get(url, params=params, timeout=15)
    print("Status code:", response.status_code)
    print(response.json())

get_power_rainfall(10.0889, 77.0595)  # Munnar