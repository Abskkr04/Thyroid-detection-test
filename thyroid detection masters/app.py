import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import warnings
warnings.filterwarnings('ignore')

pickled_model = pickle.load(open('notebooks/random_forest_model.pkl', 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Collecting form data
    features = ["age", "sex", "TSH", "T3", "T4U", "FTI", 
                "onthyroxine", "queryonthyroxine", "onantithyroidmedication",
                "sick", "pregnant", "thyroidsurgery", "I131treatment",
                "queryhypothyroid", "queryhyperthyroid", "lithium", 
                "goitre", "tumor", "hypopituitary", "psych"]
    
    values = {feature: float(request.form.get(feature, 0)) for feature in features}
    
    # Creating a DataFrame from the collected data
    df_transform = pd.DataFrame.from_dict([values])

    # Applying transformations
    df_transform['age'] = df_transform['age'] ** (1 / 2)
    df_transform['TSH'] = np.log1p(df_transform['TSH'])
    df_transform['T3'] = df_transform['T3'] ** (1 / 2)
    df_transform['T4U'] = np.log1p(df_transform['T4U'])
    df_transform['FTI'] = df_transform['FTI'] ** (1 / 2)

    # Creating a NumPy array for prediction
    arr = df_transform.values

    print("After transformation:\n")
    print(arr)

    # Making prediction
    pred = pickled_model.predict(arr)[0]

    # Mapping prediction result to a human-readable format
    if pred == 0:
        res_Val = "Hyperthyroid"
    elif pred == 1:
        res_Val = "Hypothyroid"
    else:
        res_Val = 'Negative'

    return render_template('result.html', prediction_text='Result: {}'.format(res_Val))

if __name__ == '__main__':
    app.run(debug=True, port=5000)
