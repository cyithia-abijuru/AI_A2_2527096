"""Data checks, model training, evaluation, and artifact generation."""
from pathlib import Path
import json, joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.dummy import DummyRegressor, DummyClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, silhouette_score
from data.generate_data import FEATURES, ALL_COLUMNS, verification_code, fingerprint

NUMERIC=["rainfall_mm","avg_temp_c","soil_moisture_pct","fertilizer_kg","field_size_ha","cooperative_support"]
CATEGORICAL=["season"]
RANGES={"rainfall_mm":(600,1400),"avg_temp_c":(12,26),"soil_moisture_pct":(20,80),"fertilizer_kg":(0,250),"field_size_ha":(0.2,4.0),"cooperative_support":(0,1),"yield_tons":(0.5,None)}

def validate_and_clean(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    missing_cols=sorted(set(ALL_COLUMNS)-set(raw.columns))
    if missing_cols: raise ValueError(f"Missing required columns: {missing_cols}")
    unexpected=sorted(set(raw.columns)-set(ALL_COLUMNS))
    if unexpected: raise ValueError(f"Unexpected columns: {unexpected}")
    duplicate_count=int(raw.duplicated().sum())
    missing_before={k:int(v) for k,v in raw.isna().sum().items() if v}
    df=raw.drop_duplicates().copy()
    # Reject impossible observed values; deliberate missing feature values are imputed inside pipelines.
    for col,(lo,hi) in RANGES.items():
        vals=df[col].dropna()
        if (vals < lo).any() or (hi is not None and (vals > hi).any()): raise ValueError(f"Out-of-range values in {col}")
    if not set(df['season'].dropna().unique()).issubset({'A','B'}): raise ValueError('season must be A or B')
    if not set(df['high_risk'].dropna().unique()).issubset({0,1}): raise ValueError('high_risk must be binary')
    report={"rows_before_cleaning":int(len(raw)),"rows_after_cleaning":int(len(df)),"duplicates_removed":duplicate_count,"missing_before_cleaning":missing_before,"target_leakage_columns_excluded":["yield_tons","yield per hectare","high_risk"]}
    return df.reset_index(drop=True),report

def preprocessor():
    return ColumnTransformer([("num",Pipeline([("imputer",SimpleImputer(strategy="median")),("scaler",StandardScaler())]),NUMERIC),("cat",Pipeline([("imputer",SimpleImputer(strategy="most_frequent")),("onehot",OneHotEncoder(handle_unknown="ignore"))]),CATEGORICAL)])

def train_all(df: pd.DataFrame, seed: int, out: Path, student_id: str):
    out.mkdir(parents=True,exist_ok=True); (out/'plots').mkdir(parents=True,exist_ok=True); models=out.parent/'models'; models.mkdir(parents=True,exist_ok=True)
    X=df[FEATURES].copy(); yreg=df['yield_tons'].astype(float); ycls=df['high_risk'].astype(int)
    indices=np.arange(len(df)); tr,te=train_test_split(indices,test_size=0.25,random_state=seed)
    Xtr,Xte=X.iloc[tr],X.iloc[te]; yrtr,yrte=yreg.iloc[tr],yreg.iloc[te]; yctr,ycte=ycls.iloc[tr],ycls.iloc[te]
    # Regression candidates and mean baseline.
    reg_models={"mean_baseline":Pipeline([("prep",preprocessor()),("model",DummyRegressor(strategy='mean'))]),"linear_regression":Pipeline([("prep",preprocessor()),("model",LinearRegression())]),"random_forest":Pipeline([("prep",preprocessor()),("model",RandomForestRegressor(n_estimators=160,min_samples_leaf=2,random_state=seed,n_jobs=1))])}
    reg_metrics={}; reg_preds={}
    for name,model in reg_models.items():
        model.fit(Xtr,yrtr); pred=model.predict(Xte); reg_preds[name]=pred
        reg_metrics[name]={"MAE":float(mean_absolute_error(yrte,pred)),"RMSE":float(np.sqrt(mean_squared_error(yrte,pred))),"R2":float(r2_score(yrte,pred))}
    reg_candidates=[k for k in reg_metrics if k!='mean_baseline']; best_reg=min(reg_candidates,key=lambda k:reg_metrics[k]['RMSE'])
    reg_metrics['selected_model']=best_reg; reg_metrics['selection_reason']=f"Selected {best_reg} because it had the lower held-out RMSE among candidate models."
    joblib.dump(reg_models[best_reg],models/'regression.joblib')
    # Classification candidates; class labels are excluded from all features.
    clf_models={"majority_baseline":Pipeline([("prep",preprocessor()),("model",DummyClassifier(strategy='most_frequent'))]),"logistic_regression":Pipeline([("prep",preprocessor()),("model",LogisticRegression(max_iter=1500,random_state=seed))]),"random_forest":Pipeline([("prep",preprocessor()),("model",RandomForestClassifier(n_estimators=160,min_samples_leaf=2,class_weight='balanced',random_state=seed,n_jobs=1))])}
    cls_metrics={}; cls_preds={}; cls_probs={}
    for name,model in clf_models.items():
        model.fit(Xtr,yctr); pred=model.predict(Xte); cls_preds[name]=pred
        try: probs=model.predict_proba(Xte)[:,1]
        except Exception: probs=np.zeros(len(Xte))
        cls_probs[name]=probs
        item={"precision":float(precision_score(ycte,pred,zero_division=0)),"recall":float(recall_score(ycte,pred,zero_division=0)),"F1":float(f1_score(ycte,pred,zero_division=0)),"confusion_matrix":confusion_matrix(ycte,pred,labels=[0,1]).tolist()}
        try: item['ROC_AUC']=float(roc_auc_score(ycte,probs)) if len(np.unique(ycte))==2 else None
        except ValueError: item['ROC_AUC']=None
        cls_metrics[name]=item
    best_clf=max([k for k in cls_metrics if k!='majority_baseline'],key=lambda k:cls_metrics[k]['F1'])
    cls_metrics['selected_model']=best_clf; cls_metrics['selection_reason']=f"Selected {best_clf} because it had the highest held-out F1 among candidate classifiers."
    joblib.dump(clf_models[best_clf],models/'classification.joblib')
    # Clustering: features only, scaled/imputed/encoded, no target columns.
    cluster_prep=preprocessor(); Xt=cluster_prep.fit_transform(X)
    if hasattr(Xt,'toarray'): Xt=Xt.toarray()
    scores={}; fitted={}
    for k in range(2,7):
        km=KMeans(n_clusters=k,random_state=seed,n_init=10); labels=km.fit_predict(Xt); fitted[k]=(km,labels)
        scores[str(k)]={"silhouette":float(silhouette_score(Xt,labels)),"inertia":float(km.inertia_)}
    best_k=int(max(scores,key=lambda k:scores[k]['silhouette'])); km,labels=fitted[best_k]
    profiles=[]
    for cluster in range(best_k):
        part=df.iloc[np.where(labels==cluster)[0]]
        profiles.append({"cluster":cluster,"records":int(len(part)),"mean_rainfall_mm":round(float(part.rainfall_mm.mean()),2),"mean_avg_temp_c":round(float(part.avg_temp_c.mean()),2),"mean_soil_moisture_pct":round(float(part.soil_moisture_pct.mean()),2),"mean_fertilizer_kg":round(float(part.fertilizer_kg.mean()),2),"mean_field_size_ha":round(float(part.field_size_ha.mean()),2),"interpretation":"Compare this profile with other clusters; it describes similar input conditions, not a guaranteed yield or a risk diagnosis."})
    cluster_report={"silhouette_scores_K_2_to_6":scores,"selected_K":best_k,"selection_reason":"K selected by the highest measured silhouette score among K=2..6.","profiles":profiles}
    joblib.dump({"preprocessor":cluster_prep,"model":km,"selected_K":best_k},models/'clustering.joblib')
    # Predict on test rows using selected models and export error evidence.
    reg=reg_models[best_reg]; clf=clf_models[best_clf]
    reg_test=reg.predict(Xte); clf_test=clf.predict(Xte); clf_prob=clf.predict_proba(Xte)[:,1]
    cluster_test=km.predict(cluster_prep.transform(Xte))
    pred_df=pd.DataFrame({"row_id":te,"actual_yield_tons":yrte.to_numpy(),"predicted_yield_tons":reg_test,"yield_error":yrte.to_numpy()-reg_test,"actual_high_risk":ycte.to_numpy(),"predicted_high_risk":clf_test,"risk_probability":clf_prob,"cluster":cluster_test})
    pred_df.to_csv(out/'test_predictions.csv',index=False,float_format='%.4f')
    # Required plots.
    plt.figure(figsize=(7,4)); plt.scatter(yrte,reg_test,alpha=.75); lo=min(float(yrte.min()),float(reg_test.min())); hi=max(float(yrte.max()),float(reg_test.max())); plt.plot([lo,hi],[lo,hi]); plt.xlabel('Actual yield (tons)'); plt.ylabel('Predicted yield (tons)'); plt.title('Regression residual evidence'); plt.tight_layout(); plt.savefig(out/'plots/residuals.png',dpi=140); plt.close()
    cm=confusion_matrix(ycte,clf_test,labels=[0,1]); plt.figure(figsize=(5,4)); plt.imshow(cm); plt.title('Confusion matrix'); plt.xlabel('Predicted label'); plt.ylabel('Actual label'); plt.xticks([0,1],['Not high risk','High risk']); plt.yticks([0,1],['Not high risk','High risk']);
    for (i,j),v in np.ndenumerate(cm): plt.text(j,i,str(v),ha='center',va='center')
    plt.tight_layout(); plt.savefig(out/'plots/confusion_matrix.png',dpi=140); plt.close()
    plt.figure(figsize=(7,4)); plt.scatter(Xt[:,0],Xt[:,1],c=labels,alpha=.75); plt.xlabel('Scaled feature 1'); plt.ylabel('Scaled feature 2'); plt.title(f'Farm clusters (K={best_k}; first two encoded dimensions)'); plt.tight_layout(); plt.savefig(out/'plots/clusters.png',dpi=140); plt.close()
    (out/'regression_metrics.json').write_text(json.dumps({"student_id":student_id,"verification_code":verification_code(seed),"metrics":reg_metrics},indent=2))
    (out/'classification_metrics.json').write_text(json.dumps({"student_id":student_id,"verification_code":verification_code(seed),"metrics":cls_metrics},indent=2))
    (out/'clustering_report.json').write_text(json.dumps({"student_id":student_id,"verification_code":verification_code(seed),**cluster_report},indent=2))
    return {"regression":reg_metrics,"classification":cls_metrics,"clustering":cluster_report,"test_predictions":pred_df}
