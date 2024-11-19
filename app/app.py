from flask import Flask, render_template, request, redirect
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import joblib


# Create Flask app instance
app = Flask(__name__)

# Set a secret key for Flask sessions (Important for security, especially in production)
app.secret_key = 'your_secret_key_here'  # Replace with a secure random key in 

# Data path and loading dataset
data_path = 'C:/Users/anike/project_fitness_analysis/data/gym_members_exercise_tracking.csv'  # Update to the correct path

# Load the dataset
try:
    df = pd.read_csv(data_path)
    print(df)
except FileNotFoundError:
    df = pd.DataFrame()
    print("####################Dataset not found. Please check the file path.")

# Home Route
@app.route('/')
def home():
    if 'Age' in df.columns:
        plt.figure(figsize=(10, 6))
        sns.histplot(df['Age'], kde=True, color='green')
        plt.title('Age Distribution')
        plt.xlabel('Age')
        plt.ylabel('Frequency')

        # Save the figure to the static folder
        age_image_path = 'C:/Users/anike/project_fitness_analysis/app/static/images/age_distribution.png'
        plt.savefig(age_image_path)
        plt.close()  # Close the plot to avoid it being displayed immediately
    else:
        print("Error: 'Age' column not found in DataFrame")
    
    return render_template('index.html')


# Route for Filters Page
@app.route('/filters', methods=['GET', 'POST'])
def filters():
    filtered_data = None
    if request.method == 'POST':
        workout_type = request.form.get('Workout_Type', '')
        print("Workout Type Selected:", workout_type)  # Check user input
        print("Columns in DataFrame:", df.columns)    # Check columns in DataFrame
        
        if workout_type:
            if 'Workout_Type' in df.columns:
                filtered_data = df[df['Workout_Type'].str.contains(workout_type, case=False, na=False)]
            else:
                print("Error: 'Workout_Type' column not found in DataFrame")
                
    return render_template('filters.html', data=filtered_data)

# Route for Analysis Page
@app.route('/analysis')
def analysis():
    return render_template('analysis.html')

# Route for View Analysis
@app.route('/view_analysis/<string:analysis_type>')
def view_analysis(analysis_type):
    image_path = None

    # Calories Distribution Analysis
    if analysis_type == 'calories_distribution':
        plt.figure(figsize=(10, 6))
        sns.histplot(df['Calories_Burned'], kde=True, color='blue')
        plt.title('Calories Burned Distribution')
        plt.xlabel('Calories Burned')
        plt.ylabel('Frequency')
        image_path = 'static/images/calories_distribution.png'
        plt.savefig(image_path)
        plt.close()

    # Correlation Heatmap
    elif analysis_type == 'correlation_heatmap':
        plt.figure(figsize=(10, 8))
        corr = df[['Age', 'Weight (kg)', 'Height (m)', 'BMI', 'Fat_Percentage']].corr()
        sns.heatmap(corr, annot=True, cmap='coolwarm', linewidths=0.5)
        plt.title('Correlation Heatmap of Health Metrics')
        image_path = 'static/images/correlation_heatmap.png'
        plt.savefig(image_path)
        plt.close()

    # Workout Type vs. Calories Burned
    elif analysis_type == 'workout_comparison':
        plt.figure(figsize=(12, 6))
        sns.boxplot(x='Workout_Type', y='Calories_Burned', data=df)
        plt.xticks(rotation=45)
        plt.title('Calories Burned by Workout Type')
        image_path = 'static/images/workout_comparison.png'
        plt.savefig(image_path)
        plt.close()

    # BMI vs. Fat Percentage (Health Metrics)
    elif analysis_type == 'bmi_vs_fat':
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=df, x='BMI', y='Fat_Percentage', hue='Gender')
        plt.title('BMI vs. Fat Percentage by Gender')
        plt.xlabel('BMI')
        plt.ylabel('Fat Percentage')
        image_path = 'static/images/bmi_vs_fat.png'
        plt.savefig(image_path)
        plt.close()

    
    # Water Intake vs. Calories Burned (Scatter plot)
    elif analysis_type == 'water_intake_analysis':
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=df, x='Water_Intake (liters)', y='Calories_Burned', hue='Gender')
        plt.title('Water Intake vs. Calories Burned')
        plt.xlabel('Water Intake (liters)')
        plt.ylabel('Calories Burned')
        image_path = 'static/images/water_intake_analysis.png'
        plt.savefig(image_path)
        plt.close()

    # Display the image generated based on the analysis type
    if image_path:
        return redirect(f'/{image_path}')
    else:
        return "Analysis type not found", 404
@app.route('/predict')
def predict():
    return render_template('predict.html')

# Train the regression model


# Assuming your DataFrame is already loaded into 'df'
if not df.empty:
    # Define the features (X) and target variables (y) for each model

    # Calories Burned Model
    if 'Calories_Burned' in df.columns:
        X_calories = df[['Age', 'Weight (kg)', 'Height (m)', 'BMI']]
        y_calories = df['Calories_Burned']
        X_train, X_test, y_train, y_test = train_test_split(X_calories, y_calories, test_size=0.2, random_state=42)
        model_calories = LinearRegression()
        model_calories.fit(X_train, y_train)
        joblib.dump(model_calories, 'model/calories_model.joblib')
        y_pred = model_calories.predict(X_test)
        print(f"Calories Model - Mean Squared Error: {mean_squared_error(y_test, y_pred)}")

    # Heart Rate Model
    if 'Max_BPM' in df.columns:
        X_heart_rate = df[['Age', 'Weight (kg)', 'Height (m)', 'BMI']]
        y_heart_rate = df['Max_BPM']
        X_train, X_test, y_train, y_test = train_test_split(X_heart_rate, y_heart_rate, test_size=0.2, random_state=42)
        model_heart_rate = LinearRegression()
        model_heart_rate.fit(X_train, y_train)
        joblib.dump(model_heart_rate, 'model/heart_rate_model.joblib')
        y_pred = model_heart_rate.predict(X_test)
        print(f"Heart Rate Model - Mean Squared Error: {mean_squared_error(y_test, y_pred)}")

    
else:
    print("Dataset is empty or missing necessary columns. Models not trained.")


@app.route('/predict_calories', methods=['GET', 'POST'])
def predict_calories():
    prediction = None
    if request.method == 'POST':
        # Get user input
        age = float(request.form.get('age', 0))
        weight = float(request.form.get('weight', 0))
        height = float(request.form.get('height', 0))
        bmi = float(request.form.get('bmi', 0))

        # Load the saved model
        model_path = 'model/calories_model.joblib'
        if os.path.exists(model_path):
            model = joblib.load(model_path)
            # Make a prediction
            prediction = model.predict([[age, weight, height, bmi]])[0]
        else:
            print("Error: Model file not found.")
    
    return render_template('predict_calories.html', prediction=prediction)

@app.route('/predict_heart_rate', methods=['GET', 'POST'])
def predict_heart_rate():
    prediction = None
    if request.method == 'POST':
        try:
            # Get user input from the form
            age = float(request.form.get('age', 0))
            weight = float(request.form.get('weight', 0))
            height = float(request.form.get('height', 0))
            bmi = float(request.form.get('bmi', 0))

            # Load the saved heart rate model
            model_path = 'model/heart_rate_model.joblib'
            if os.path.exists(model_path):
                model = joblib.load(model_path)
                # Make a prediction
                prediction = model.predict([[age, weight, height, bmi]])[0]
            else:
                print("Error: Heart rate model file not found.")
        except Exception as e:
            print(f"Error: {e}")
    
    return render_template('predict_heart_rate.html', prediction=prediction)


    


if __name__ == '__main__':
    app.run(debug=True)
