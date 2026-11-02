import pandas as pd
import numpy as np
import joblib, os, requests

from sdv.single_table import GaussianCopulaSynthesizer    ## core learning engine
from sdv.metadata import Metadata     ## schema detector

from faker import Faker  ## Generate Synthetic Student Names

import warnings
warnings.filterwarnings("ignore", message="X does not have valid feature names")

lr_pipeline = joblib.load('lr_pipeline.pkl')
trained_cols = lr_pipeline.feature_names_in_

NUMERIC_COLS = ['daily_study_hours', 'previous_score', 'attendance_percentage', 'sleep_hours']
OBJECT_COLS  = ['student_id', 'student_name']

def get_synthetic_student_data(
        raw_csv_filepath='raw_data.csv', 
        metadata_filepath='students_metadata.json'
    ):

    ## Load raw student data (locally stored)
    raw_df = pd.read_csv(raw_csv_filepath)

    ## include specific columns
    df = raw_df[OBJECT_COLS + NUMERIC_COLS].copy()

    if os.path.exists(metadata_filepath):
        metadata = Metadata.load_from_json(metadata_filepath)

    else:
        metadata = Metadata.detect_from_dataframe(df)    ## scans dataframe and figures out datatype of columns
        metadata.update_column(     ## don't learn/ synthesize to student_id (primary key)
            column_name='student_id',
            sdtype='id'
        )
        metadata.update_column(     ## don't learn/ synthesize to student_name
            column_name='student_name',
            sdtype='id'
        )
        metadata.set_primary_key('student_id')
        metadata.save_to_json(metadata_filepath)   ## we won't recompute metadata every time

    synthesizer = GaussianCopulaSynthesizer(metadata)
    synthesizer.fit(df)  ## learns statistical information about each row
    ## Range, Common Value, Data Distribution and Correlation between Columns

    synthetic_df = synthesizer.sample(num_rows=20)    ## create 20 synthetic data

    ## Overwrite student_id  (S1001 - S1020)
    synthetic_df['student_id'] = [f'S{i:04d}' for i in range(1001, 1021)]
    ## Overwrite student names
    fake = Faker('en_IN')
    synthetic_df['student_name'] = [fake.name() for _ in range(20)]

    return synthetic_df

def upper_hours_threshold(col):
    if col=='daily_study_hours':
        return 16
    else:
        return 14

def validate_dtypes(df: pd.DataFrame):
    invalid_values, missing_values = [], None

    if df.empty:
        invalid_values.append("DataFrame is empty, no data submitted")
        return invalid_values

    for col in OBJECT_COLS + NUMERIC_COLS:
        if col not in df.columns:
            invalid_values.append(f'Missing column: {col}')

    # 1. Object columns
    for col in OBJECT_COLS:
        if col not in df.columns:
            continue

        if df[col].dtype not in ['object', 'string']:
            invalid_values.append(f"'{col}' must be text, got dtype: {df[col].dtype}")

    # 2. Numeric columns — dtype FIRST, range SECOND
    for col in NUMERIC_COLS:
        if col not in df.columns:
            continue

        # Step 1: Check dtype first
        converted = pd.to_numeric(df[col], errors='coerce')
        invalid_rows = df[converted.isna() & df[col].notna()]
        if not invalid_rows.empty:
            bad_vals = invalid_rows[col].tolist()
            invalid_values.append(f"'{col}' must be numeric, got invalid values: {bad_vals}")
            continue    

        # Step 2: Only check range if dtype is valid
        upper_limit = upper_hours_threshold(col) if 'hours' in col else 100
        lower_limit = 4 if col=='sleep_hours' else 0
        unreal_mask = (df[col] > upper_limit) | (df[col] < lower_limit)
        if unreal_mask.any():
            unreal_values = df[col][unreal_mask].tolist()
            msg = "unrealistic hour values" if 'hours' in col else "unrealistic academic data"
            invalid_values.append(f"'{col}' contains {msg}: {unreal_values}")

    if invalid_values:
        return invalid_values, missing_values
    
    # 3. Validate total hours spend studying and sleeping
    if 'daily_study_hours' in df.columns and 'sleep_hours' in df.columns:
        total_hours = df['daily_study_hours'] + df['sleep_hours']
        over_limit = total_hours[total_hours>20]
        if not over_limit.empty:
            invalid_values.append(f'Study + sleep hours exceeds 20 in some students: {over_limit.tolist()}')

    # 4. Display warning message for missing values
    if df.isna().any().any():
        missing_values = 'Cannot predict final score for student with missing values'

    return invalid_values, missing_values

def predict_student_score(df: pd.DataFrame):
    # 1. Identify which rows have ANY null values
    is_null = df[OBJECT_COLS + NUMERIC_COLS].isna().any(axis=1)
    
    # 2. Handle rows with missing data (Vectorized string assignment)
    # We use .loc to target only the rows where is_null is True
    df.loc[is_null, 'final_score_pred'] = df.loc[is_null].apply(
        lambda x: f'Cannot predict final score for {x["student_name"]}. Missing Values: {" ".join(x.index[x.isna()])} in submitted excel',
        axis=1
    )
    df.loc[is_null, 'result_pred'] = 'Could not classify'

    # 3. Handle rows with complete data
    # Create a mask for complete rows
    complete = ~is_null
    if complete.any():
        valid_df = df.loc[complete].copy()
        
        # Vectorized calculations
        valid_df['idle_hours'] = 24.0 - (valid_df['sleep_hours'] + valid_df['daily_study_hours'])
        valid_df['engagement_score'] = valid_df['daily_study_hours'] * (valid_df['attendance_percentage'] / 100)
        valid_df['well_rested'] = (valid_df['sleep_hours']>=7).astype('int')
        valid_df['is_motivated'] = ((valid_df['previous_score']>=65) & (valid_df['engagement_score']>=valid_df['engagement_score'].median())).astype('int')
        valid_df['at_risk'] = ((valid_df['idle_hours']>14) & (valid_df['well_rested']==0)).astype('int')
        
        # Vectorized Prediction
        # Most pipelines (like sklearn) handle a whole DataFrame at once
        preds = lr_pipeline.predict(valid_df[trained_cols])
        valid_df['final_score_pred'] = preds.clip(0, 100)
        
        # Vectorized Classification
        valid_df['result_pred'] = np.where(valid_df['final_score_pred'] >= 50, 'Pass', 'Fail')
        
        # Merge results back into the original dataframe
        df.update(valid_df)

    return df

def format_hours(val):
    if pd.isna(val):
        return 'N/A'
    h, m = divmod(round(val* 60), 60)
    return f'{h}hr {m}min'

def format_percentages(val):
    if pd.isna(val) or not isinstance(val, (int, float)):
        return 'N/A'
    return f'{val:.2f}%'

def student_feedback(row: pd.Series):
    flags = []

    if not pd.isna(row['daily_study_hours']) and row['daily_study_hours'] < 2:
        flags.append('Student has fewer studying hours, may require academic guidance and motivation')
    if not pd.isna(row['previous_score']) and row['previous_score'] < 60:
        flags.append(f"Student has weak fundamentals, scored {row['previous_score']:.2f}% at their previous exam, consider remedial support")
    if not pd.isna(row['attendance_percentage']) and row['attendance_percentage'] < 75:
        flags.append('Attendance is below 75%, consider following up with student')
    if not pd.isna(row['sleep_hours']) and row['sleep_hours'] < 6:
        flags.append('Student shows lack of sleep, may affect their concentration and academic performance')

    return flags

def get_student_feedback_reports(df: pd.DataFrame):
    student_report = {}
    
    pred_df = predict_student_score(df)
    
    for _, row in pred_df.iterrows():
        
        improv_feedbacks = student_feedback(row)
        
        if row['student_id'] not in student_report:
            student_report[row['student_id']] = {
                ## Student Name
                'name': row['student_name'],

                ## Academic
                'previous_score': format_percentages(row['previous_score']),
                'attendance': format_percentages(row['attendance_percentage']),

                ## LifeStyle
                'study_hours': format_hours(row['daily_study_hours']),
                'sleep_hours': format_hours(row['sleep_hours']),

                ## Predictions
                'final_score': format_percentages(row['final_score_pred']),
                'result': row['result_pred'],

                ## Feedbacks
                'feedbacks': [row['final_score_pred']] if isinstance(row['final_score_pred'], str) else (improv_feedbacks or None)
            }

    return student_report

def generate_html_report(student_report: dict):
    tables_html = ""
    students = list(student_report.items())

    for i, (student_id, student_data) in enumerate(students):
        feedbacks = student_data.get('feedbacks', '')
        if feedbacks:
            feedback_html = "".join(f'<li>{fb}</li>' for fb in feedbacks)
            feedback_html = f'<ul>{feedback_html}</ul>'
        else:
            feedback_html = '<span class="great">✅ Doing Great. Keep up the Consistency!</span>'

        result = student_data.get('result', '').upper()
        result_class = 'pass' if result == 'PASS' else 'fail'

        # Page break after every 2 students
        page_break = 'style="page-break-after: always;"' if (i + 1) % 2 == 0 else ''

        tables_html += f"""
        <div class="student-block" {page_break}>
            <table>
                <thead>
                    <tr>
                        <th class="student-id">Student ID: {student_id}</th>
                        <th class="student-name">Student Name: {student_data.get('name', '')}</th>
                    </tr>
                    <tr>
                        <th class="section-header">📚 Academic</th>
                        <th class="section-header">🏃 Lifestyle</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td><span class="label">Previous Score</span>{student_data.get('previous_score', '')}</td>
                        <td><span class="label">Study Hrs/Day</span>{student_data.get('study_hours', '')}</td>
                    </tr>
                    <tr>
                        <td><span class="label">Attendance</span>{student_data.get('attendance', '')}</td>
                        <td><span class="label">Sleep Hrs/Day</span>{student_data.get('sleep_hours', '')}</td>
                    </tr>
                    <tr class="summary-row">
                        <td><b>Predicted Score:</b> {student_data.get('final_score', '')}</td>
                        <td class="{result_class}"><b>Result: {result}</b></td>
                    </tr>
                    <tr class="feedback-row">
                        <td colspan="2">
                            <span class="feedback-label">Improvement Feedback</span>
                            {feedback_html}
                        </td>
                    </tr>
                </tbody>
            </table>
        </div>
        """

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Student Report</title>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600&display=swap');

        * {{ margin: 0; padding: 0; box-sizing: border-box; }}

        body {{
            font-family: 'DM Sans', sans-serif;
            font-size: 13px;
            color: #2d2d2d;
            padding: 28px;
            background: #fff;
        }}

        .student-block {{
            margin-bottom: 24px;
        }}

        table {{
            width: 100%;
            border-collapse: collapse;
            border: 1px solid #d0d7de;
            border-radius: 8px;
            overflow: hidden;
        }}

        th, td {{
            padding: 9px 14px;
            border: 1px solid #d0d7de;
            vertical-align: middle;
        }}

        /* Student ID header */
        th.student-id {{
            background-color: #1a5276;
            color: #ffffff;
            font-size: 14px;
            font-weight: 600;
            letter-spacing: 0.4px;
            text-align: left;
        }}

        /* Student Name header */
        th.student-name {{
            background-color: #1a5276;
            color: #ffffff;
            font-size: 14px;
            font-weight: 600;
            letter-spacing: 0.4px;
            text-align: left;
        }}

        /* Section headers */
        th.section-header {{
            background-color: #eaf0fb;
            color: #1a3a5c;
            font-weight: 600;
            text-align: center;
            font-size: 12.5px;
        }}

        /* Data rows */
        tbody tr:nth-child(odd) {{ background-color: #f9fbfd; }}
        tbody tr:nth-child(even) {{ background-color: #ffffff; }}

        .label {{
            display: block;
            font-size: 10.5px;
            color: #7a8a99;
            font-weight: 500;
            margin-bottom: 1px;
        }}

        .empty-cell {{
            color: #bbb;
            text-align: center;
        }}

        /* Summary row */
        .summary-row td {{
            background-color: #f0f4f8;
            font-size: 13px;
        }}

        .pass {{ color: #1e7e45; font-weight: 600; text-align: center; }}
        .fail {{ color: #c0392b; font-weight: 600; text-align: center; }}

        /* Feedback row */
        .feedback-row td {{
            background-color: #fafafa;
            padding: 10px 14px;
        }}

        .feedback-label {{
            display: block;
            font-weight: 600;
            font-size: 11.5px;
            color: #555;
            margin-bottom: 4px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}

        .feedback-row ul {{
            margin: 0;
            padding-left: 18px;
            color: #444;
        }}

        .feedback-row li {{
            margin-bottom: 3px;
            line-height: 1.5;
        }}

        .great {{
            color: #1e7e45;
            font-style: italic;
        }}
    </style>
</head>
<body>
    {tables_html}
</body>
</html>"""

class HTMLEmail:
    def send_html_content(self, to_email, html_content, subject, from_email, brevo_api_key):
        """A method to send html email at the given address"""
        try:
            if not from_email or not brevo_api_key:
                return None, 'Brevo credentials not configured'
            
            response = requests.post(
                'https://api.brevo.com/v3/smtp/email',
                headers={
                    'api-key': brevo_api_key,
                    'Content-Type': 'application/json'
                },
                json={
                    'sender': {'email': from_email},
                    'to': [{'email': to_email}],
                    'subject': subject,
                    'htmlContent': html_content
                }
            )

            if response.status_code == 201:
                return 'Email sent successfully', None
            else:
                error = f'Brevo error: {response.status_code} - {response.text}'
                return None, error
            
        except Exception as exc:
            error = f'Failed to send email:\n{exc}'
            return None, error