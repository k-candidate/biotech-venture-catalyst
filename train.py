#!/usr/bin/env python


import pickle
import re

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline


def load_and_engineer_features(file_path='data/clinical_trials_raw.csv'):
    """Ingests raw clinical trial metadata and transforms columns into engineered features."""
    print("Loading raw data matrix...")
    df = pd.read_csv(file_path)
    
    print("Running text and temporal feature transformations...")
    # Parse dates and calculate exact operational lifespans
    df['start_dt'] = pd.to_datetime(df['start_date'], errors='coerce', format='mixed')
    df['complete_dt'] = pd.to_datetime(df['complete_date'], errors='coerce', format='mixed')
    df['start_year'] = df['start_dt'].dt.year.fillna(-1).astype(int)
    df['start_month'] = df['start_dt'].dt.month.fillna(-1).astype(int)
    df['duration_days'] = (df['complete_dt'] - df['start_dt']).dt.days.fillna(-1).astype(int)
    
    # Extract structural markers from nested indication codes
    df['num_conditions'] = df['icdcodes'].apply(lambda x: len(str(x).split('__')) if pd.notnull(x) else 0)
    df['primary_disease_chapter'] = df['icdcodes'].apply(lambda x: str(x)[0] if pd.notnull(x) and len(str(x)) > 0 else 'Unknown')
    
    # Extract proxy text density metrics from unstructured protocols
    df['criteria_word_count'] = df['eligibility_criteria'].apply(lambda x: len(str(x).split()) if pd.notnull(x) else 0)
    df['exclusion_keyword_count'] = df['eligibility_criteria'].apply(
        lambda x: sum(1 for _ in re.finditer(r'(?i)\b(exclude|exclusion|no history|prior concurrent)\b', str(x))) if pd.notnull(x) else 0
    )
    
    # Isolate strictly the targeted active modeling features
    features_to_use = [
        'start_year', 'start_month', 'duration_days', 'num_conditions', 
        'primary_disease_chapter', 'criteria_word_count', 'exclusion_keyword_count', 'Y'
    ]
    return df[features_to_use].copy()


def prepare_data_splits(df):
    """Splits data into train/val/test, matching the exact order of operations from research."""
    n = len(df)
    n_val = int(n * 0.2)
    n_test = int(n * 0.2)
    n_train = n - n_val - n_test
    
    # Generate identical random mask index arrays using fixed seed 42
    idx = np.arange(n)
    np.random.seed(42)
    np.random.shuffle(idx)
    
    df_shuffled = df.iloc[idx].reset_index(drop=True)
    
    # Carve into standard MLZoomcamp tabular partitions
    df_train = df_shuffled.iloc[:n_train].copy()
    df_val = df_shuffled.iloc[n_train:n_train+n_val].copy()
    df_test = df_shuffled.iloc[n_train+n_val:].copy()
    
    # Recombine train & validation for the ultimate production dataset pool
    df_full_train = pd.concat([df_train, df_val]).reset_index(drop=True)
    
    # Extract standard numpy array target vectors
    y_full_train = df_full_train['Y'].values
    y_test = df_test['Y'].values
    
    # Drop target columns from operational feature dataframes to generate dictionary logs
    df_full_train = df_full_train.drop(columns=['Y']).reset_index(drop=True)
    df_test = df_test.drop(columns=['Y']).reset_index(drop=True)
    
    full_train_dicts = df_full_train.to_dict(orient='records')
    test_dicts = df_test.to_dict(orient='records')
    
    return full_train_dicts, y_full_train, test_dicts, y_test


def build_and_train_pipeline(train_dicts, y_train):
    """Assembles and trains the cohesive scikit-learn pipeline structure."""
    print("Assembling scikit-learn preprocessing and model pipeline...")
    pipeline = Pipeline([
        ('vectorizer', DictVectorizer(sparse=False)),
        ('classifier', RandomForestClassifier(
            n_estimators=50,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        ))
    ])
    pipeline.fit(train_dicts, y_train)
    return pipeline


def evaluate_pipeline(pipeline, test_dicts, y_test):
    """Verifies that the production model generalizes properly on unseen test sets."""
    y_pred_test = pipeline.predict_proba(test_dicts)[:, 1]
    auc_score = roc_auc_score(y_test, y_pred_test)
    print(f"Verified Final Test Hold-out ROC AUC: {auc_score:.5f}")


def serialize_pipeline(pipeline, output_file='model.bin'):
    """Saves the fully integrated training pipeline to disk as a binary file."""
    print(f"Exporting pipeline model artifact to: {output_file}")
    with open(output_file, 'wb') as f_out:
        pickle.dump(pipeline, f_out)


def main():
    print("Starting production model training script pipeline...")
    
    # Step 1: Feature Engineering Layer
    df_model_ready = load_and_engineer_features()
    
    # Step 2: Extract Data Dictionary Partitions
    full_train_dicts, y_full_train, test_dicts, y_test = prepare_data_splits(df_model_ready)
    
    # Step 3: Fit the Production Pipeline Asset
    production_pipeline = build_and_train_pipeline(full_train_dicts, y_full_train)
    
    # Step 4: Final Generalization Validation Check
    evaluate_pipeline(production_pipeline, test_dicts, y_test)
    
    # Step 5: Serialize Binary Artifact
    serialize_pipeline(production_pipeline)
    
    print("train.py successfully executed. Training completed successfully.")


if __name__ == "__main__":
    main()