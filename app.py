import os
import cv2
import numpy as np
import streamlit as st
from PIL import Image
import pydicom  # Import pydicom for DICOM file support
from streamlit_chat import message  # Import the message component for chat
import requests  # To make HTTP requests to aiXplain API
import time  # For polling delay
# Set the API key for aiXplain
API_KEY = '799b8640ed5d2e45959f34bc3adf4f4c45515d0d492d171b8e7f07cd0da48c1e'
API_URL = 'https://models.aixplain.com/api/v1/execute/64788e666eb56313aa7ebac1'  # aiXplain model URL

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

# Set page configuration
st.set_page_config(page_title="MammoCare", page_icon="🩺", layout="centered")

# Add CSS styles for better formatting
st.markdown(
    """
    <style>
    body {
        background-color: #f4f4f4;
    }
    .hero {
        background-image: url('https://preview.free3d.com/img/2021/12/3190213534600398410/2yo1z0c2.jpg');
        background-size: cover;
        background-position: center;
        height: 800px;
        border-radius:20px;
        display: flex;
        justify-content: center;
        align-items: center;
        text-align: center;
        color: white;
    }
    .cta-button {
        background-color: #E91E63;
        padding: 10px 20px;
        color: white;
        border-radius: 5px;
        text-decoration: none;
        font-weight: bold;
    }
    .cta-button:hover {
        background-color: #D81B60;
    }
    .section {
        padding: 50px;
        text-align: center;
    }
    .feature-box {
        background-color: white;
        padding: 20px;
        border-radius: 5px;
        box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Sidebar navigation
st.sidebar.title("MammoCare")
pages = st.sidebar.radio("Navigate", ["Home", "Mammogram manual App", "Auto", "About Us", "Contact"])

# Functions for mammogram processing
def load_image(file_path):
    return cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

def preprocess_image(image, blur_kernel_size):
    equalized_image = cv2.equalizeHist(image)
    blurred_image = cv2.GaussianBlur(equalized_image, (blur_kernel_size, blur_kernel_size), 0)
    return blurred_image

def remove_pectoral_muscle(image, side, start, end):
    x1, y1 = start
    x2, y2 = end
    mask = np.zeros_like(image)
    
    cv2.line(mask, (y1, x1), (y2, x2), 255, thickness=6)
    
    if side == "Left":
        if y1 < y2:
            points = np.array([[0, 0], [0, image.shape[0]], [y2, x2], [y1, x1]])
        else:
            points = np.array([[y1, x1], [y2, x2], [image.shape[1], image.shape[0]], [image.shape[1], 0]])
    else:  # Right
        if y1 < y2:
            points = np.array([[y1, x1], [y2, x2], [image.shape[1], image.shape[0]], [image.shape[1], 0]])
        else:
            points = np.array([[0, 0], [0, image.shape[0]], [y2, x2], [y1, x1]])

    cv2.fillPoly(mask, [points], 255)
    image[mask == 255] = 0
    
    return image

def find_highest_dense_region(image):
    _, thresholded_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresholded_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        return thresholded_image, image

    largest_contour = max(contours, key=cv2.contourArea)
    dense_mask = np.zeros_like(image)
    cv2.drawContours(dense_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
    
    highest_dense_image = cv2.bitwise_and(image, image, mask=dense_mask)
    dense_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(dense_image, [largest_contour], -1, (0, 0, 255), 2)

    return thresholded_image, dense_image

def load_dicom(file):
    dicom_data = pydicom.dcmread(file)
    image = dicom_data.pixel_array  
    if np.max(image) > 255:
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return image

def poll_aixplain_model(request_id):
    poll_url = f"https://models.aixplain.com/api/v1/data/{request_id}"
    
    while True:
        response = requests.get(poll_url, headers=headers)
        if response.status_code == 200:
            result = response.json()
            if result.get('completed', False):  
                return result.get('data', 'No result data available')
            else:
                time.sleep(5)  
        else:
            return f"Error: Failed to poll the job. Status code: {response.status_code}"

def query_aixplain_model(user_input):
    data = {
        'text': user_input
    }
    
    try:
        response = requests.post(API_URL, headers=headers, json=data)
        if response.status_code == 201:
            result = response.json()
            request_id = result.get('requestId')
            if request_id:
                return poll_aixplain_model(request_id)  
        else:
            return f"Error: API request failed with status code {response.status_code}: {response.text}"
    except Exception as e:
        return f"Exception occurred: {e}"

# Main content of the application
if pages == "Home":
    # Header
    st.markdown("<h1 style='text-align: center; color: #E91E63;'>MammoCare</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>Using AI to enhance breast cancer detection and diagnosis.</p>", unsafe_allow_html=True)

    # Hero section
    st.markdown("""<div class="hero"><div><h2>Advanced Mammogram Image Processing</h2><p>Leverage the power of AI to detect breast cancer early and accurately.</p><a href="#features" class="cta-button">Explore Features</a></div></div>""", unsafe_allow_html=True)

    # About section
    st.markdown("<div class='section' id='about'><h2>About MammoCare</h2><p>MammoCare is a state-of-the-art mammogram image processing platform, leveraging AI to improve early breast cancer detection and diagnostic accuracy.</p></div>", unsafe_allow_html=True)

    # Features section
    st.markdown("<div class='section' id='features'><h2>Key Features</h2></div>", unsafe_allow_html=True)
    st.markdown("<div class='feature-box'><h3>AI-Powered Analysis</h3><p>Automatically analyze mammogram images to detect abnormalities with high accuracy.</p></div>", unsafe_allow_html=True)
    st.markdown("<div class='feature-box'><h3>Image Enhancement</h3><p>Enhance image clarity for better interpretation by medical professionals.</p></div>", unsafe_allow_html=True)
    st.markdown("<div class='feature-box'><h3>Data Security</h3><p>Secure and confidential processing of patient data, adhering to HIPAA regulations.</p></div>", unsafe_allow_html=True)

    # # Footer
    # st.markdown("<footer style='text-align: center; padding: 20px; background-color: #E91E63; color: white;'>© 2024 MammoCare. All rights reserved.</footer>", unsafe_allow_html=True)

elif pages == "Mammogram manual App":
    st.title("Mammogram Image Processing")
    # Show Instructions at the top left and expand by default
    with st.expander("Instructions", expanded=True):
        st.markdown("""    
        ### How to Use the Application:
        1. **Upload Image**: Upload a mammogram image using the uploader. Supported formats: PNG, JPG, JPEG, DICOM (.dcm).
        2. **Adjust Settings**: Use the sliders and options to customize the processing:
            - **Apply Equalization**: Enhance contrast.
            - **Blur Kernel Size**: Control the level of blurring to reduce noise.
            - **Side Selection**: Specify the side of the breast.
            - **Start & End Coordinates**: Define the region for pectoral muscle removal.
        3. **Display Settings**: Choose which images to display after processing.
        4. **Chatbot**: Use the chatbot to ask questions about mammogram analysis.
        """)
    
    # Upload DICOM or Image File
    uploaded_file = st.file_uploader("Upload a DICOM file or a mammogram image", type=["dcm", "jpg", "jpeg", "png"])

    if uploaded_file is not None:
        if uploaded_file.name.endswith('.dcm'):
            image = load_dicom(uploaded_file)
        else:
            image = Image.open(uploaded_file)
            image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Display uploaded image
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Uploaded Image", use_column_width=True)

        # Preprocessing options
        st.sidebar.header("Image Preprocessing Options")
        blur_kernel_size = st.sidebar.slider("Gaussian Blur Kernel Size", 1, 15, 5, 2)

        # Preprocess the image
        preprocessed_image = preprocess_image(image, blur_kernel_size)
        with col2:
            st.image(preprocessed_image, caption="Preprocessed Image", use_column_width=True)

        # Pectoral muscle removal options
        st.sidebar.header("Pectoral Muscle Removal")
        side = st.sidebar.selectbox("Select Side", ["Left", "Right"])
        start_point = st.sidebar.slider("Start Point (X, Y)", 0, image.shape[1]-1, (0, 0), 1)
        end_point = st.sidebar.slider("End Point (X, Y)", 0, image.shape[1]-1, (image.shape[1]-1, image.shape[0]-1), 1)

        # Remove pectoral muscle from image
        muscle_removed_image = remove_pectoral_muscle(preprocessed_image.copy(), side, start_point, end_point)

        # Show muscle removed image side by side with the original
        col1, col2 = st.columns(2)
        with col1:
            st.image(muscle_removed_image, caption="Muscle Removed Image", use_column_width=True)

        # High-density region detection
        thresholded_image, highest_dense_image = find_highest_dense_region(muscle_removed_image)
        
        with col2:
            st.image(thresholded_image, caption="High Density Region", use_column_width=True)

        # Show highest dense region image side by side
        col1, col2 = st.columns(2)
        with col1:
            st.image(highest_dense_image, caption="Highest Dense Region", use_column_width=True)

        # User input for AI model
        user_input = st.text_input("Enter your diagnosis request or question for the AI model")
        if st.button("Submit"):
            response = query_aixplain_model(user_input)
            st.success(response) 

elif pages == "Auto":
    st.title("Auto")
    st.markdown("""
    <p>MammoCare is dedicated to revolutionizing the early detection of breast cancer. Our team comprises medical professionals and AI experts who work together to enhance diagnostic accuracy.</p>
    """, unsafe_allow_html=True)                      

elif pages == "About Us":
    st.title("About Us")
    st.markdown("""
    <p>MammoCare is dedicated to revolutionizing the early detection of breast cancer. Our team comprises medical professionals and AI experts who work together to enhance diagnostic accuracy.</p>
    """, unsafe_allow_html=True)

elif pages == "Contact":
    st.title("Contact Us")
    st.markdown("""
    <p>If you have any questions or need support, please reach out to us at <a href="mailto:support@mammocare.com">support@mammocare.com</a>.</p>
    """, unsafe_allow_html=True)



# Footer
st.markdown("<footer style='text-align: center; padding: 20px; background-color: #E91E63; color: white;'>© 2024 MammoCare. All rights reserved.</footer>", unsafe_allow_html=True)
