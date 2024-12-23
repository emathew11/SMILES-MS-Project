'''Pick up model. If config file exists, load the trained model.'''

from pathlib import Path
import pickle as pk
from model import build_model

class ModelService:
    def __init__(self):
        self.model = None

    def load_model(self, model_name='name'):
        model_path = Path(f'models/{model_name}')

        if not model_path.exists():
            build_model()

        self.model = pk.load(open(f'models/{model_name}', 'rb'))

    def predict(self, input_parameters):
        return self.model.predict([input_parameters])
    
