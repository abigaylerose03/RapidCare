import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

text = ""
app=Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))


@app.route('/')
def home():
	return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
	int_features = [float(x) for x in request.form.values()]
	final_features = [np.array(int_features)]
	prediction = model.predict(final_features)
	output = round(prediction[0], 2)
	if output == 1:
		text = "benign"
	elif output == 0:
		text = "malignant"
	else:
		return "error!"

	return render_template('index.html', prediction_text='Based on this information, the tumor(s) should be ' + text)

@app.route('/results', methods=['POST'])
def results():
	data = request.get_json(force=True)
	prediction = model.predict([np.array(list(data.values))])
	output = prediction[0]
	return jsonify(output)

if __name__ == "__main__":
	app.run(debug=True)
