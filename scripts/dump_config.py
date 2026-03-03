import joblib
import json
art=joblib.load('models/score_models.joblib')
print(json.dumps(art.get('config'), indent=2))
