import streamlit as st
from ultralytics import YOLO
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

# Load the YOLOv8 model
model = YOLO('model.pt')

# Streamlit app
def main():
    st.title("Image Prediction App")
    st.write("Upload an image to get predictions!")

    # File uploader for image input
    uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Load the image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image",  use_container_width=True)
        st.write("Processing the image...")

        # Save the uploaded file temporarily for YOLO input
        temp_path = "temp_image.jpg"
        image.save(temp_path)

        # Perform prediction
        results = model.predict(
            source=temp_path,
            conf=0.25
        )

        # Display the predictions
        for result in results:
            # Convert result.plot() (numpy array) to an image
            predicted_img = result.plot()
            predicted_img = Image.fromarray(predicted_img)
            st.image(predicted_img, caption="Predicted Image",  use_container_width=True)

    # Define questions for specific problems
    questions_mapping = {
        "acne": [
            "How severe is your acne? (1: Mild, 2: Moderate, 3: Severe)",
            "Do you experience pain or redness with your acne? (1: Yes, 2: No)",
            "Do you get frequent breakouts? (1: Yes, 2: No)",
        ],
        "dark_circles": [
            "Are your dark circles hereditary? (1: Yes, 2: No)",
            "Do you experience puffiness with your dark circles? (1: Yes, 2: No)",
            "Do your dark circles get worse with lack of sleep? (1: Yes, 2: No)",
        ],
        "wrinkles": [
            "Do you moisturize daily? (1: Yes, 2: No)",
            "Do you use anti-aging products? (1: Yes, 2: No)",
            "Do you have frequent sun exposure? (1: Yes, 2: No)",
        ],
    }

    # Define recommendations with descriptions
    recommendations_mapping = {
        "acne": {
            "mild": [
                {
                    "ingredient": "Salicylic Acid", 
                    "description": "This ingredient gently exfoliates the skin, unclogs pores, and prevents breakouts. Look for cleansers or spot treatments with salicylic acid to manage mild acne."
                },
                {
                    "ingredient": "Niacinamide", 
                    "description": "Known for its anti-inflammatory properties, niacinamide reduces redness and balances oil production. Opt for serums or moisturizers containing niacinamide."
                },
            ],
            "moderate": [
                {
                    "ingredient": "Benzoyl Peroxide", 
                    "description": "Effective in killing acne-causing bacteria, benzoyl peroxide reduces inflammation and clears moderate acne. Use it in spot treatments or cleansers."
                },
                {
                    "ingredient": "Azelaic Acid", 
                    "description": "Azelaic acid helps to unblock pores, reduce swelling, and fade post-acne marks. Consider creams or gels with this ingredient for visible improvements."
                },
            ],
            "severe": [
                {
                    "ingredient": "Retinoids", 
                    "description": "Retinoids promote skin cell turnover and reduce severe acne and scarring. Look for prescription-strength or over-the-counter products labeled as retinol or tretinoin."
                },
                {
                    "ingredient": "Clindamycin", 
                    "description": "Clindamycin is an antibiotic that targets bacteria causing severe acne. This is typically found in medicated creams prescribed by dermatologists."
                },
            ],
        },
        "dark_circles": {
            "hereditary": [
                {
                    "ingredient": "Vitamin K", 
                    "description": "Vitamin K improves blood circulation around the eyes and lightens hereditary dark circles. Choose eye creams with this ingredient for long-term results."
                },
            ],
            "puffiness": [
                {
                    "ingredient": "Caffeine", 
                    "description": "Caffeine helps constrict blood vessels and reduces puffiness. Look for eye serums or roll-ons with caffeine for instant refreshment."
                },
            ],
            "sleep_related": [
                {
                    "ingredient": "Hyaluronic Acid", 
                    "description": "Hyaluronic acid hydrates and plumps the under-eye area, improving the appearance of dark circles caused by dehydration or lack of sleep. Opt for eye creams or masks with hyaluronic acid."
                },
            ],
        },
        "wrinkles": {
            "moisturize": [
                {
                    "ingredient": "Hyaluronic Acid", 
                    "description": "Hyaluronic acid deeply hydrates and plumps the skin, softening fine lines and wrinkles. Look for lightweight serums or rich moisturizers with this ingredient."
                },
            ],
            "anti_aging": [
                {
                    "ingredient": "Peptides", 
                    "description": "Peptides boost collagen production and improve skin elasticity. Anti-aging creams or serums with peptides can make your skin firmer and smoother."
                },
            ],
            "sun_exposure": [
                {
                    "ingredient": "Sunscreen", 
                    "description": "Sunscreen protects against harmful UV rays that accelerate aging. Use a broad-spectrum SPF 30 or higher every day to prevent and reduce wrinkles."
                },
            ],
        },
    }

    # Load the trained YOLO model
    model = YOLO('model.pt')

    # Run YOLO for detection
    results = model.predict(source=image_path, conf=0.1, iou=0.01)

    # Parse YOLO detections and remove duplicates
    detected_classes = [result.names[int(box[-1])] for result in results for box in result.boxes.data.tolist()]
    detected_classes = list(set(detected_classes))  # Remove duplicates

    # Display unique detected issues
    print("\nDetected Issues:")
    for idx, issue in enumerate(detected_classes, 1):
        print(f"{idx}: {issue}")

    # Let the user select issues
    selected_issues = input("\nEnter the numbers corresponding to the issues you want to target (comma-separated): ")
    selected_issues = [detected_classes[int(idx) - 1] for idx in selected_issues.split(",")]

    # Ask targeted questions
    user_responses = {}
    for issue in selected_issues:
        print(f"\nQuestions for {issue}:")
        for question in questions_mapping[issue]:
            answer = input(question + " ")
            user_responses[issue] = user_responses.get(issue, []) + [int(answer)]

    # Generate recommendations based on user responses
    final_recommendations = []
    for issue in selected_issues:
        responses = user_responses[issue]
        if issue == "acne":
            severity = ["mild", "moderate", "severe"][responses[0] - 1]
            final_recommendations.extend(recommendations_mapping["acne"][severity])
        elif issue == "dark_circles":
            if responses[0] == 1:  # Hereditary
                final_recommendations.extend(recommendations_mapping["dark_circles"]["hereditary"])
            if responses[1] == 1:  # Puffiness
                final_recommendations.extend(recommendations_mapping["dark_circles"]["puffiness"])
            if responses[2] == 1:  # Sleep-related
                final_recommendations.extend(recommendations_mapping["dark_circles"]["sleep_related"])
        elif issue == "wrinkles":
            if responses[0] == 1:  # Moisturize
                final_recommendations.extend(recommendations_mapping["wrinkles"]["moisturize"])
            if responses[1] == 1:  # Anti-aging products
                final_recommendations.extend(recommendations_mapping["wrinkles"]["anti_aging"])
            if responses[2] == 1:  # Sun exposure
                final_recommendations.extend(recommendations_mapping["wrinkles"]["sun_exposure"])

    # Remove duplicates and display final recommendations with descriptions
    final_recommendations = {rec["ingredient"]: rec["description"] for rec in final_recommendations}
    print("\nBased on your inputs, we recommend looking for products containing the following ingredients:")
    for ingredient, description in final_recommendations.items():
        print(f"- {ingredient}: {description}")


if __name__ == "__main__":
    main()
