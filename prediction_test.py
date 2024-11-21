

import requests



url = 'http://localhost:9696/predict'



data = { "hour": 6.0, 
        "temp": 11.5, "pres": 1010.5, "wd": 180.0, "wspm": 0.5, 
        "pm25_avg": 52.5, "o3_avg": 8.62}

response = requests.post(url, json=data).json()
print(response)





