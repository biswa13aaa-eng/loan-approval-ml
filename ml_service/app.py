"""HTTP inference layer around the existing saved scikit-learn pipelines."""
from __future__ import annotations
import json, sys
from functools import lru_cache
from pathlib import Path
from typing import Any
import joblib, pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sklearn.metrics import confusion_matrix
ROOT = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(ROOT))
from src.config import CATEGORICAL_COLS, ID_COL, MODELS_DIR, NUMERICAL_COLS, RAW_DATA_FILE, TARGET_COL
from src.data_loader import split_data
app = FastAPI(title="LOANWISE AI ML Service", docs_url=None, redoc_url=None)
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173"], allow_methods=["*"], allow_headers=["*"])
PATHS = {"Logistic Regression": MODELS_DIR / "logistic_regression.joblib", "Random Forest": MODELS_DIR / "random_forest.joblib", "XGBoost": MODELS_DIR / "xgboost.joblib"}
class PredictionRequest(BaseModel):
    model_name: str
    inputs: dict[str, Any]
@lru_cache
def dataset(): return pd.read_csv(RAW_DATA_FILE)
@lru_cache
def models(): return {name: joblib.load(path) for name, path in PATHS.items() if path.exists()}
@lru_cache
def artifacts():
    results = ROOT / "reports" / "results"; answer = {}
    for key, name in [("comparison", "model_comparison.csv"), ("importance", "feature_importance.csv")]:
        if (results / name).exists(): answer[key] = pd.read_csv(results / name)
    if (MODELS_DIR / "model_metadata.json").exists(): answer["metadata"] = json.loads((MODELS_DIR / "model_metadata.json").read_text())
    return answer
def records(frame): return json.loads(frame.to_json(orient="records"))
@app.get("/health")
def health(): return {"success": True, "data": {"status": "operational", "models_loaded": list(models())}}
@app.get("/dataset/summary")
def summary():
    frame=dataset(); inputs=[x for x in frame.columns if x not in {ID_COL,TARGET_COL}]
    return {"success":True,"data":{"rows":len(frame),"columns":len(frame.columns),"input_features":len(inputs),"numerical_features":NUMERICAL_COLS,"categorical_features":CATEGORICAL_COLS,"target":TARGET_COL,"approved":int((frame[TARGET_COL]=="Y").sum()),"rejected":int((frame[TARGET_COL]=="N").sum()),"missing_values":int(frame.isna().sum().sum()),"duplicate_rows":int(frame.duplicated().sum()),"columns_list":frame.columns.tolist(),"target_distribution":[{"status":k,"count":int(v)} for k,v in frame[TARGET_COL].value_counts().items()]}}
@app.get("/dataset/preview")
def preview(page:int=1,limit:int=15,search:str=""):
    frame=dataset(); limit=min(max(limit,1),100); page=max(page,1)
    if search: frame=frame[frame.astype(str).apply(lambda r:r.str.contains(search,case=False,na=False).any(),axis=1)]
    start=(page-1)*limit
    return {"success":True,"data":{"rows":records(frame.iloc[start:start+limit]),"total":len(frame),"page":page,"limit":limit}}
@app.get("/models/performance")
def performance(): return {"success":True,"data":{"models":records(artifacts()["comparison"]),"metadata":artifacts().get("metadata",{})}}
@app.get("/models/{model_name}/confusion-matrix")
def matrix(model_name:str):
    model=models().get(model_name)
    if not model: raise HTTPException(404,"Requested model is unavailable")
    _, test, _, target=split_data(dataset()); values=confusion_matrix(target,model.predict(test),labels=[0,1]).tolist()
    return {"success":True,"data":{"model_name":model_name,"matrix":values,"labels":["Rejected","Approved"]}}
@app.get("/features/importance")
def importance(): return {"success":True,"data":records(artifacts()["importance"])}
@app.post("/predict")
def predict(request:PredictionRequest):
    model=models().get(request.model_name); required=NUMERICAL_COLS+CATEGORICAL_COLS
    if not model: raise HTTPException(404,"Requested model is unavailable")
    if missing:=[field for field in required if field not in request.inputs]: raise HTTPException(422,f"Missing required fields: {', '.join(missing)}")
    try:
        frame=pd.DataFrame([{field:request.inputs[field] for field in required}]); outcome=int(model.predict(frame)[0]); probability=float(model.predict_proba(frame)[0,1]) if hasattr(model,"predict_proba") else None
    except Exception: raise HTTPException(422,"Unable to generate prediction from the provided inputs")
    risk=None if probability is None else ("Low Risk" if probability>=.70 else "Moderate Risk" if probability>=.50 else "High Risk")
    return {"success":True,"data":{"prediction":"Approved" if outcome else "Rejected","approved":bool(outcome),"approval_probability":probability,"rejection_probability":None if probability is None else 1-probability,"model_name":request.model_name,"risk_level":risk}}
