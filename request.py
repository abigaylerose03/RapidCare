import requests 

url = "http://localhost:5000/results"
r = requests.post(url, json={'mean radius': 10.39, 'mean texture': 10.25, 'mean smoothness': 0.3252, 'mean compactness': 0.23353, 
	'mean symmetry': 0.435232, 'mean fractal dimension': 0.135, 'radius error': 0.2235, 'texture error': 0.2354, 'smoothness error': 0.3351,
	'compactness error': 0.2355, 'symmetry error': 0.25323, 'fractal dimension error': 0.232533})

print(r.json())

# mean radius', 'mean texture', 'mean smoothness', 'mean compactness',
#        'mean symmetry', 'mean fractal dimension', 'radius error',
#        'texture error', 'smoothness error', 'compactness error',
#        'symmetry error', 'fractal dimension error'],
#       dtype='object')