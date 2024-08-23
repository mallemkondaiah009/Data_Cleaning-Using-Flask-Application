from flask import Flask, render_template, request, send_file, abort
import pandas as pd
import numpy as np
import os

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
CLEANED_FOLDER = 'cleaned'

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

if not os.path.exists(CLEANED_FOLDER):
    os.makedirs(CLEANED_FOLDER)

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files.get('file')
        if file:
            file_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(file_path)

            # Verify the file is saved correctly
            if not os.path.isfile(file_path):
                return "Failed to save file. Please try again."

            # Load the dataset
            try:
                if file.filename.endswith('.csv'):
                    df = pd.read_csv(file_path)
                elif file.filename.endswith('.xlsx'):
                    df = pd.read_excel(file_path)
                else:
                    return "Unsupported file format"
            except Exception as e:
                return f"Error loading file: {e}"

            # Apply the data cleaning steps
            try:
                df_cleaned = clean_data(df)
            except Exception as e:
                return f"Error cleaning data: {e}"

            # Save the cleaned data
            cleaned_file_path = os.path.join(CLEANED_FOLDER, 'cleaned_' + file.filename)
            try:
                if file.filename.endswith('.csv'):
                    df_cleaned.to_csv(cleaned_file_path, index=False)
                elif file.filename.endswith('.xlsx'):
                    df_cleaned.to_excel(cleaned_file_path, index=False)
            except Exception as e:
                return f"Error saving cleaned file: {e}"

            return send_file(cleaned_file_path, as_attachment=True)

    return render_template('index.html')
def clean_data(df):
    # Step 1: Handling Missing Values
    df = df.copy()

    # Drop columns with more than 50% missing values
    df.dropna(thresh=df.shape[0]*0.5, axis=1, inplace=True)

    # Fill numeric columns with interpolation
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    df[numeric_columns] = df[numeric_columns].interpolate(method='linear', limit_direction='forward', axis=0)

    # If there are still missing values after interpolation, fill them with the median
    df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].median())

    # Fill categorical columns with 'Unknown' or a placeholder
    categorical_columns = df.select_dtypes(include=['object']).columns
    df[categorical_columns] = df[categorical_columns].fillna('Unknown')

    # Drop rows with any remaining missing values
    df.dropna(inplace=True)

    # Step 2: Remove Duplicates
    df.drop_duplicates(inplace=True)

    # Step 3: Convert Data Types
    # Convert any object columns that look like numbers to numeric
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='ignore')

    # Step 4: Standardize Text Data
    for col in categorical_columns:
        df[col] = df[col].str.strip().str.replace(r'\s+', ' ', regex=True)

    # Step 5: Handle Outliers (using IQR method)
    for col in numeric_columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        df = df[~((df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR)))]

    # Step 6: Rename Columns
    df.columns = [col.strip().lower().replace(' ', '_') for col in df.columns]

    return df


if __name__ == '__main__':
    app.run(debug=True)
