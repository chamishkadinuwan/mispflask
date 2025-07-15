import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from flask import Flask, jsonify, send_file
from flask_cors import CORS
import os

# Flask app setup
app = Flask(__name__)
CORS(app)

def estimate_resources(people):
    return {
        'Water Liters': int(people * 3),
        'Food Packs': int(people * 2),
        'Medical Kits': max(1, int(people) // 10),
        'Blankets': int(people),
        'Sanitation Kits': max(1, int(people) // 5)
    }

def run_prediction():
    df = pd.read_excel("data/Flood_Disaster_Data.xlsx")

    df['Date'] = pd.to_datetime(df['Date'])
    df['Disaster Type'] = df['Disaster Type'].str.lower()
    df['Flood'] = (df['Disaster Type'] == 'flood').astype(int)

    le = LabelEncoder()
    df['City'] = df['City'].astype(str)
    df['City_Code'] = le.fit_transform(df['City'])

    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day
    df['DayOfYear'] = df['Date'].dt.dayofyear
    df['Week'] = df['Date'].dt.isocalendar().week

    features = ['Rainfall (mm)', 'Month', 'Day', 'DayOfYear', 'Week', 'City_Code']
    X = df[features]
    y_class = df['Flood']
    y_reg = df['# of Affected People']

    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X, y_class, stratify=y_class, test_size=0.2, random_state=42)
    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X, y_reg, test_size=0.2, random_state=42)

    clf = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42)
    clf.fit(X_train_c, y_train_c)

    reg = RandomForestRegressor(n_estimators=200, random_state=42)
    reg.fit(X_train_r, y_train_r)

    next_year = df['Date'].max().year + 1
    cities = df['City'].unique()
    next_dates = pd.date_range(start=f'{next_year}-01-01', end=f'{next_year}-12-31', freq='D')
    future = pd.MultiIndex.from_product([next_dates, cities], names=['Date', 'City']).to_frame(index=False)

    future['Month'] = future['Date'].dt.month
    future['Day'] = future['Date'].dt.day
    future['DayOfYear'] = future['Date'].dt.dayofyear
    future['Week'] = future['Date'].dt.isocalendar().week
    future['Rainfall (mm)'] = np.random.randint(0, 500, size=len(future))
    future['City_Code'] = le.transform(future['City'])

    future_X = future[features]
    future['Flood_Predicted'] = clf.predict(future_X)
    future['Affected_People_Predicted'] = reg.predict(future_X)

    predicted_floods = future[future['Flood_Predicted'] == 1]
    os.makedirs("output", exist_ok=True)
    predicted_floods.to_csv("output/predicted_flood_days.csv", index=False)

    resource_rows = []
    for _, row in future.iterrows():
        if row['Affected_People_Predicted'] > 0:
            resources = estimate_resources(row['Affected_People_Predicted'])
            resource_rows.append({
                'Date': row['Date'],
                'City': row['City'],
                'Predicted Affected People': int(row['Affected_People_Predicted']),
                **resources
            })

    resource_df = pd.DataFrame(resource_rows)
    resource_df.to_csv("output/resource_needs_2026.csv", index=False)

    # Grouped monthly summary
    resource_df['Month'] = pd.to_datetime(resource_df['Date']).dt.to_period('M').astype(str)
    monthly_df = resource_df.groupby(['Month', 'City']).agg({
        'Predicted Affected People': 'sum',
        'Water Liters': 'sum',
        'Food Packs': 'sum',
        'Medical Kits': 'sum',
        'Blankets': 'sum',
        'Sanitation Kits': 'sum'
    }).reset_index()
    monthly_df.to_csv("output/resource_needs_monthly_2026.csv", index=False)

    monthly_counts = predicted_floods['Date'].dt.month.value_counts().sort_index()
    plt.figure(figsize=(10, 5))
    sns.barplot(x=monthly_counts.index, y=monthly_counts.values, palette='Blues_d')
    plt.xlabel('Month')
    plt.ylabel('Predicted Flood Days')
    plt.title(f'Predicted Flood Occurrence by Month in {next_year}')
    plt.xticks(ticks=np.arange(0, 12), labels=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                                               'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    plt.tight_layout()
    plt.savefig("output/flood_prediction_chart.png")
    plt.close()

run_prediction()

@app.route('/api/flood-predictions')
def get_predictions():
    df = pd.read_csv("output/predicted_flood_days.csv")
    return jsonify(df.to_dict(orient='records'))

@app.route('/api/resource-needs')
def get_resource_needs():
    df = pd.read_csv("output/resource_needs_2026.csv")
    return jsonify(df.to_dict(orient='records'))

@app.route('/api/resource-needs-monthly')
def get_resource_needs_monthly():
    df = pd.read_csv("output/resource_needs_monthly_2026.csv")
    return jsonify(df.to_dict(orient='records'))

@app.route('/api/flood-chart')
def get_chart():
    return send_file("output/flood_prediction_chart.png", mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True)
