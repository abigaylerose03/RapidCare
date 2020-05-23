import requests 

url = "http://localhost:5000/results"
r = requests.post(url, json={'radius': 10, 'smoothness': 10, 'symmetry': 1})

print(r.json())
