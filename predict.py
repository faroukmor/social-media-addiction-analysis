import sys
import joblib
import pandas as pd

def get_float_input(prompt_text):
    """Safely capture float input without crashing on invalid types."""
    while True:
        try:
            return float(input(prompt_text))
        except ValueError:
            print("--> Error: Invalid input. Please enter a valid numerical value.\n")

def main():
    print("====================================================")
    print("    Social Media Addiction Prediction System (CLI)   ")
    print("====================================================\n")
    
    # 1. Load the pre-trained pipeline model
    try:
        model = joblib.load('social_media_addiction_model.joblib')
    except FileNotFoundError:
        print("Error: Model file 'social_media_addiction_model.joblib' not found.")
        print("Please ensure the model file is in the same directory.")
        sys.exit(1)
        
    print("Model loaded successfully. Please enter the requested metrics:\n")
    
    # 2. Collect user inputs safely using the helper function
    internet_penetration = get_float_input("1. Internet Penetration Rate (e.g., 0.85): ")
    mental_support       = get_float_input("2. Mental Health Support Index (e.g., 50.0): ")
    gdp_index            = get_float_input("3. Country GDP Index (e.g., 1.2): ")
    youth_ratio          = get_float_input("4. Youth Population Ratio (e.g., 0.25): ")
    baseline_pressure    = get_float_input("5. Baseline Addiction Pressure Score (e.g., 3.0): ")
    tiktok_min           = get_float_input("6. Daily TikTok Usage in Minutes (e.g., 120): ")
    instagram_min        = get_float_input("7. Daily Instagram Usage in Minutes (e.g., 90): ")
    night_ratio          = get_float_input("8. Night Usage Ratio (e.g., 0.40 for 40%): ")
    scroll_velocity      = get_float_input("9. Scroll Velocity Score (e.g., 4.5): ")
    attention_span       = get_float_input("10. Attention Span Score (e.g., 5.0): ")
    dopamine_dep         = get_float_input("11. Dopamine Dependency Score (e.g., 6.5): ")
    impulsivity          = get_float_input("12. Impulsivity Index (e.g., 2.5): ")
    sleep_hours          = get_float_input("13. Sleep Hours Per Night (e.g., 7.0): ")
    sleep_quality        = get_float_input("14. Sleep Quality Index (e.g., 6.0): ")
        
    # 3. Structure inputs matching the exact feature names seen during fit time
    input_data = pd.DataFrame([{
        'internet_penetration': internet_penetration,
        'mental_health_support_index': mental_support,
        'gdp_index': gdp_index,
        'youth_population_ratio': youth_ratio,
        'baseline_addiction_pressure': baseline_pressure,
        'tiktok_minutes_daily': tiktok_min,
        'instagram_minutes_daily': instagram_min,
        'night_usage_ratio': night_ratio,
        'scroll_velocity': scroll_velocity,
        'attention_span_score': attention_span,
        'dopamine_dependency_score': dopamine_dep,
        'impulsivity_index': impulsivity,
        'sleep_hours': sleep_hours,
        'sleep_quality_index': sleep_quality
    }])
    
    # 4. Execute prediction and extract probabilities
    prediction = model.predict(input_data)[0]
    probabilities = model.predict_proba(input_data)[0]
    classes = model.classes_
    
    # 5. Display the final dynamic diagnosis
    print("\n=================== DIAGNOSIS ===================")
    print(f" Predicted Addiction Level: {prediction}")
    print("-------------------------------------------------")
    print(" Class Probabilities:")
    for cls, prob in zip(classes, probabilities):
        print(f"  - {cls}: {prob*100:.2f}%")
    print("=================================================")

if __name__ == "__main__":
    main()