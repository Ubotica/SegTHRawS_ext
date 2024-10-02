import os

thraws_data_path = os.path.join(os.path.dirname(__file__),'data','THRAWS')


DATASETS_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)),'data','datasets')
os.makedirs(DATASETS_PATH,exist_ok=True)

models_path = ""
