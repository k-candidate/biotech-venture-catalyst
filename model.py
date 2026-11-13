#!/usr/bin/env python


import re
from typing import Any

import pandas as pd
from pydantic import BaseModel, Field


class RawTrialInput(BaseModel):
    # In Pydantic v2, omit the ellipsis (...) and use keyword-only arguments to stay type-safe.
    # 'example' is now passed via 'examples' as a list.
    start_date: str = Field(description="Raw start date text (e.g., 'March 1996')", examples=["March 1996"])
    complete_date: str | None = Field(default=None, description="Raw completion date text (can be null for terminated trials)", examples=["July 2000"])
    icdcodes: str = Field(description="Double-underscore delimited ICD-10 keys", examples=["D89.810__C95.91"])
    eligibility_criteria: str = Field(description="Unstructured free text clinical protocol rules", examples=["Age 13 and over. Exclude pregnancy."])

def transform_raw_to_features(raw_input: RawTrialInput) -> dict[str, Any]:
    """
    Validate the raw input and derive the features used by the model.
    """
    raw_dict = raw_input.model_dump()
    record: dict[str, Any] = {}

    # 1. Temporal Feature Extraction
    start_dt = pd.to_datetime(raw_dict['start_date'], errors='coerce', format='mixed')
    complete_dt = pd.to_datetime(raw_dict['complete_date'], errors='coerce', format='mixed')
    
    record['start_year'] = int(start_dt.year) if pd.notnull(start_dt) else -1
    record['start_month'] = int(start_dt.month) if pd.notnull(start_dt) else -1
    
    if pd.notnull(start_dt) and pd.notnull(complete_dt):
        record['duration_days'] = int((complete_dt - start_dt).days)
    else:
        record['duration_days'] = -1

    # 2. ICD-10 Medical Feature Extraction
    icd_str = raw_dict['icdcodes']
    record['num_conditions'] = len(icd_str.split('__')) if icd_str else 0
    record['primary_disease_chapter'] = str(icd_str)[0] if icd_str and len(str(icd_str)) > 0 else 'Unknown'

    # 3. Unstructured Eligibility Criteria Text Mining
    criteria_text = raw_dict['eligibility_criteria'] or ""
    record['criteria_word_count'] = len(criteria_text.split())
    record['exclusion_keyword_count'] = sum(1 for _ in re.finditer(r'(?i)\b(exclude|exclusion|no history|prior concurrent)\b', criteria_text))

    return record
