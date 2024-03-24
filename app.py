from flask import Flask, render_template, request
import plotly.graph_objs as go
import plotly.express as px
import numpy as np

import logging
import flask_monitoringdashboard as dashboard

from keras.models import load_model

from src.get_data import GetData
from src.utils import create_figure, prediction_from_model 

app = Flask(__name__)

# Configuration/initialisation du monitoring
dashboard.config.init_from(file='config.cfg')

# Configuration du journal de logging
logging.basicConfig(filename='error.log', level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')

data_retriever = GetData(url="https://data.rennesmetropole.fr/api/explore/v2.1/catalog/datasets/etat-du-trafic-en-temps-reel/exports/json?lang=fr&timezone=Europe%2FBerlin&use_labels=true&delimiter=%3B")
data = data_retriever()

model = load_model('model.h5') 

# Gestionnaire d'erreurs global (pour les erreurs non gérées)
@app.errorhandler(Exception)
def handle_error(error):
    logging.exception(f'Unhandled exception: {error}', exc_info=True)
    return 'Internal Server Error', 500


@app.route('/', methods=['GET', 'POST'])

def index():
    try : 
        if request.method == 'POST':
            
            fig_map = create_figure(data)
            graph_json = fig_map.to_json()

            selected_hour = request.form['hour']

            cat_predict = prediction_from_model(model, selected_hour)

            color_pred_map = {0:["Prédiction : Libre", "green"], 1:["Prédiction : Dense", "orange"], 2:["Prédiction : Bloqué", "red"]}

            return render_template('index.html', graph_json=graph_json, text_pred=color_pred_map[cat_predict][0], color_pred=color_pred_map[cat_predict][1])

        else:

            fig_map = create_figure(data)
            graph_json = fig_map.to_json()

            return render_template('index.html', graph_json=graph_json)

    except Exception as e:
        logging.exception(f"An error occurred: {str(e)}")
        return "Internal Server Error", 500

# Monitoring de l'app
dashboard.bind(app) 

if __name__ == '__main__':
    app.run(debug=True)
