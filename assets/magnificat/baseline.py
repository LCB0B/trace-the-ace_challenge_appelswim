""""""
from __future__ import annotations
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from .features import STRUCTURED_COLS
TEXT_COL = 'transcript_text'

def _to_dense(x):
    return x

def build_pipeline(max_features: int=50000, svd_components: int=200, random_state: int=42) -> Pipeline:
    """"""
    from sklearn.decomposition import TruncatedSVD
    text_branch = Pipeline([('tfidf', TfidfVectorizer(max_features=max_features, ngram_range=(1, 2), min_df=3, sublinear_tf=True)), ('svd', TruncatedSVD(n_components=svd_components, random_state=random_state))])
    pre = ColumnTransformer([('text', text_branch, TEXT_COL), ('num', FunctionTransformer(_to_dense, validate=False), STRUCTURED_COLS)])
    clf = HistGradientBoostingClassifier(learning_rate=0.08, max_iter=400, max_leaf_nodes=63, l2_regularization=1.0, early_stopping=True, validation_fraction=0.1, random_state=random_state)
    return Pipeline([('pre', pre), ('clf', clf)])

def build_xgb_pipeline(max_features: int=50000, min_df: int=3, n_estimators: int=600, learning_rate: float=0.05, max_depth: int=6, subsample: float=0.8, colsample_bytree: float=0.5, reg_lambda: float=2.0, min_child_weight: float=5.0, n_jobs: int=-1, random_state: int=42) -> Pipeline:
    """"""
    from xgboost import XGBClassifier
    text_branch = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2), min_df=min_df, sublinear_tf=True)
    pre = ColumnTransformer([('text', text_branch, TEXT_COL), ('num', FunctionTransformer(_to_dense, validate=False), STRUCTURED_COLS)], sparse_threshold=1.0)
    clf = XGBClassifier(objective='binary:logistic', eval_metric='logloss', tree_method='hist', n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth, subsample=subsample, colsample_bytree=colsample_bytree, reg_lambda=reg_lambda, min_child_weight=min_child_weight, n_jobs=n_jobs, random_state=random_state)
    return Pipeline([('pre', pre), ('clf', clf)])

def build_meta_pipeline(C: float=1.0, max_features: int=5000, ngram_range: tuple=(1, 2), sublinear_tf: bool=False, max_iter: int=2000, random_state: int=42) -> Pipeline:
    """"""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import OneHotEncoder
    pre = ColumnTransformer([('obj_text', TfidfVectorizer(max_features=max_features, ngram_range=ngram_range, sublinear_tf=sublinear_tf), 'learning_objective'), ('obj_id', OneHotEncoder(handle_unknown='ignore'), ['learning_objective_id'])], sparse_threshold=1.0)
    clf = LogisticRegression(C=C, max_iter=max_iter, random_state=random_state)
    return Pipeline([('pre', pre), ('clf', clf)])

def build_meta_gbm_pipeline(max_features: int=5000, n_estimators: int=300, learning_rate: float=0.05, num_leaves: int=31, min_child_samples: int=50, reg_lambda: float=5.0, n_jobs: int=-1, random_state: int=42) -> Pipeline:
    """"""
    from lightgbm import LGBMClassifier
    from sklearn.preprocessing import OneHotEncoder
    pre = ColumnTransformer([('obj_text', TfidfVectorizer(max_features=max_features, ngram_range=(1, 2)), 'learning_objective'), ('obj_id', OneHotEncoder(handle_unknown='ignore'), ['learning_objective_id'])], sparse_threshold=1.0)
    clf = LGBMClassifier(objective='binary', metric='binary_logloss', n_estimators=n_estimators, learning_rate=learning_rate, num_leaves=num_leaves, min_child_samples=min_child_samples, reg_lambda=reg_lambda, n_jobs=n_jobs, random_state=random_state, verbose=-1)
    return Pipeline([('pre', pre), ('clf', clf)])

def build_meta_dialogue_gbm_pipeline(dialogue_cols, max_features: int=5000, n_estimators: int=300, learning_rate: float=0.05, num_leaves: int=31, min_child_samples: int=50, reg_lambda: float=5.0, n_jobs: int=-1, random_state: int=42) -> Pipeline:
    """"""
    from lightgbm import LGBMClassifier
    from sklearn.preprocessing import OneHotEncoder
    pre = ColumnTransformer([('obj_text', TfidfVectorizer(max_features=max_features, ngram_range=(1, 2)), 'learning_objective'), ('obj_id', OneHotEncoder(handle_unknown='ignore'), ['learning_objective_id']), ('dlg', 'passthrough', list(dialogue_cols))], sparse_threshold=1.0)
    clf = LGBMClassifier(objective='binary', metric='binary_logloss', n_estimators=n_estimators, learning_rate=learning_rate, num_leaves=num_leaves, min_child_samples=min_child_samples, reg_lambda=reg_lambda, n_jobs=n_jobs, random_state=random_state, verbose=-1)
    return Pipeline([('pre', pre), ('clf', clf)])

def build_meta_calibrated_pipeline(method: str='isotonic', cv: int=3, random_state: int=42):
    """"""
    from sklearn.calibration import CalibratedClassifierCV
    return CalibratedClassifierCV(build_meta_pipeline(random_state=random_state), method=method, cv=cv)

def build_lgbm_pipeline(max_features: int=20000, min_df: int=3, n_estimators: int=400, learning_rate: float=0.05, num_leaves: int=63, subsample: float=0.8, colsample_bytree: float=0.5, reg_lambda: float=2.0, min_child_samples: int=20, n_jobs: int=-1, random_state: int=42) -> Pipeline:
    """"""
    from lightgbm import LGBMClassifier
    text_branch = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2), min_df=min_df, sublinear_tf=True)
    pre = ColumnTransformer([('text', text_branch, TEXT_COL), ('num', FunctionTransformer(_to_dense, validate=False), STRUCTURED_COLS)], sparse_threshold=1.0)
    clf = LGBMClassifier(objective='binary', metric='binary_logloss', n_estimators=n_estimators, learning_rate=learning_rate, num_leaves=num_leaves, subsample=subsample, subsample_freq=1, colsample_bytree=colsample_bytree, reg_lambda=reg_lambda, min_child_samples=min_child_samples, n_jobs=n_jobs, random_state=random_state, verbose=-1)
    return Pipeline([('pre', pre), ('clf', clf)])