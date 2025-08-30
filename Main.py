import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Set page configuration for wide layout
st.set_page_config(page_title="Mahogany Plant Disease Recognition System", layout="wide")

# Tensorflow Model Prediction
def model_prediction(test_image):
    model = tf.keras.models.load_model("trained_plant_disease_model.keras")
    
    # Convert uploaded image to the appropriate size
    image = test_image.resize((128, 128))  # Resize to match model input size
    
    # Convert image to RGB if it has an alpha channel
    if image.mode == 'RGBA':
        image = image.convert('RGB')
    
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # Convert single image to batch
    predictions = model.predict(input_arr)
    return np.argmax(predictions)  # Return index of max element


# Inject custom CSS for animations (optional)
st.markdown("""
    <style>
    .animated-text:hover {
        animation: grow 1s ease-in-out infinite;
    }
    @keyframes grow {
        0% {transform: scale(1);}
        50% {transform: scale(1.2);}
        100% {transform: scale(1);}
    }
    </style>
""", unsafe_allow_html=True)

# Title for the dashboard
st.title("🌿 Mahogany Plant Disease Recognition System 🌿")

# Use columns to make content more spread across the screen
col1, col2, col3 = st.columns([1, 6, 1])  # Adjust the ratio for wider middle section

with col2:  # This ensures all content is centered but stretched to full width
    # Create tabs for navigation with animations
    tab1, tab2, tab3 = st.tabs(["🌱 Home", "📜 About", "🔬 Disease Recognition"])

    # Home Tab with Animation
    with tab1:
        st.header("🌼 Home Page 🌼")
        image_path = r"C:\Users\priya\OneDrive\Desktop\plant_disease_predication\home_page.jpeg"
        st.image(image_path, use_column_width=True)

        # Adding a plant-themed animation (balloons)
        st.balloons()

        st.markdown("""
        Welcome to the **Mahogany Plant Disease Recognition System!** 🌿🔍
        
        Our mission is to help in identifying plant diseases efficiently. Upload an image of a plant, and our system will analyze it to detect any signs of diseases. Together, let's protect our crops and ensure a healthier harvest!

        ### How It Works
        1. **Upload Image:** Go to the **Disease Recognition** tab and upload an image of a plant with suspected diseases.
        2. **Analysis:** Our system will process the image using advanced algorithms to identify potential diseases.
        3. **Results:** View the results and recommendations for further action.

        ### Why Choose Us?
        - **Accuracy:** Our system utilizes state-of-the-art machine learning techniques for accurate disease detection.
        - **User-Friendly:** Simple and intuitive interface for seamless user experience.
        - **Fast and Efficient:** Receive results in seconds, allowing for quick decision-making.

        ### Get Started
        Click on the **Disease Recognition** tab to upload an image and experience the power of our Plant Disease Recognition System!

        ### About Us
        Learn more about the project, our team, and our goals on the **About** tab.
        """)

    # About Tab with Snow Animation
    with tab2:
        st.header("About 🍃")
        
        # Adding snow animation
        st.snow()
        
        st.markdown("""
                    #### About Dataset
                    This dataset is recreated using offline augmentation from the original dataset. The original dataset can be found on this GitHub repo.
                    This dataset consists of about 87K RGB images of healthy and diseased crop leaves, categorized into 38 different classes. The total dataset is divided into an 80/20 ratio of training and validation sets, preserving the directory structure.
                    A new directory containing 33 test images is created later for prediction purposes.
                    
                    #### Content
                    1. Train (70,295 images)
                    2. Test (33 images)
                    3. Validation (17,572 images)
                    """)

    # Disease Recognition Tab with Animation on Prediction
    with tab3:
        st.header("Disease Recognition 🌱")
        
        # File uploader for disease prediction
        test_image = st.file_uploader("Choose an Image:", type=["jpg", "png", "jpeg"])
        
        if test_image is not None:
            # Open the uploaded image file
            image = Image.open(test_image)
            
            # Set a smaller image preview size (e.g., 300 pixels in width)
            st.image(image, caption="Uploaded Image", width=300)  # Set custom width for preview
            
            # Adding a fun spinner animation
            with st.spinner("Analyzing the disease..."):
                if st.button("Predict"):
                    # Adding an animation for fun
                    st.snow()  # Show snow effect for the prediction
                    result_index = model_prediction(image)

                    # List of possible class names
                    class_name = ['bark_eating_caterpillar', 'gummosis', 'leaf_eating_caterpillar', 'shootborer']

                    # Adding success notification with animation
                    st.success(f"Model predicts it's {class_name[result_index]} 🌟")
                    st.balloons()  # Celebratory balloons for the prediction
