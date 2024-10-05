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

# Set the page configuration for the theme
st.set_page_config(
    page_title="Pectoral Muscle Removal & Dense Region Detection",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load and preprocess the mammogram image
def load_image(file_path):
    return cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

def preprocess_image(image, blur_kernel_size):
    # Apply histogram equalization to enhance contrast
    equalized_image = cv2.equalizeHist(image)
    # Apply Gaussian blur with user-controlled kernel size to reduce noise
    blurred_image = cv2.GaussianBlur(equalized_image, (blur_kernel_size, blur_kernel_size), 0)
    return blurred_image

# Remove pectoral muscle region
def remove_pectoral_muscle(image, side, start, end):
    x1, y1 = start
    x2, y2 = end
    mask = np.zeros_like(image)
    
    # Draw a diagonal line on the mask
    cv2.line(mask, (y1, x1), (y2, x2), 255, thickness=6)
    
    # Create a polygon for the area to remove
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

# Find the most dense region (largest contour)
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

# Function to load and process DICOM files
def load_dicom(file):
    dicom_data = pydicom.dcmread(file)
    image = dicom_data.pixel_array  # Extract pixel data
    # Normalize pixel data to 0-255 if necessary
    if np.max(image) > 255:
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return image

# Function to poll aiXplain model for results
def poll_aixplain_model(request_id):
    """Poll the AI model for results using the request ID."""
    poll_url = f"https://models.aixplain.com/api/v1/data/{request_id}"
    
    while True:
        response = requests.get(poll_url, headers=headers)
        if response.status_code == 200:
            result = response.json()
            if result.get('completed', False):  # Check if the job is complete
                return result.get('data', 'No result data available')
            else:
                time.sleep(5)  # Wait before checking again
        else:
            return f"Error: Failed to poll the job. Status code: {response.status_code}"

# Function to query the aiXplain model
def query_aixplain_model(user_input):
    """Send a request to the aiXplain model and handle the response."""
    data = {
        'text': user_input
    }
    
    try:
        response = requests.post(API_URL, headers=headers, json=data)
        if response.status_code == 201:  # Job accepted, but not complete
            result = response.json()
            request_id = result.get('requestId')
            if request_id:
                return poll_aixplain_model(request_id)  # Poll for result
        else:
            return f"Error: API request failed with status code {response.status_code}: {response.text}"
    except Exception as e:
        return f"Exception occurred: {e}"

# Streamlit app with enhanced sidebar and layout
def main():
    # Custom CSS for dark theme with light pink elements
    st.markdown(
        """
        <style>
        body {
            background-color: #1e1e1e;  /* Dark Background */
        }
        .sidebar .sidebar-content {
            background-color: #1e1e1e;  /* Dark Sidebar */
        }
        .stButton>button {
            background-color: #FF69B4;  /* Hot Pink for buttons */
            color: white;
        }
        .stTextInput>div>input {
            background-color: #FFFAFA;  /* Light Pink for text input */
            color: #000000;  /* Black text */
        }
        .stSlider>div>label {
            color: #FFFFFF;  /* White for slider labels */
        }
        .stCheckbox>div>label {
            color: #FFFFFF;  /* White for checkbox labels */
        }
        .stRadio>div>label {
            color: #FFFFFF;  /* White for radio button labels */
        }
        .stSelectbox>div>label {
            color: #FFFFFF;  /* White for select box labels */
        }
        .stSidebar>div>label {
            color: #dcdcdc;  /* Dull white for sidebar header */
        }
        h1 {
            color: #dcdcdc;  /* Dull white for main header */
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.title("Pectoral Muscle Removal & Dense Region Detection")

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

    # Sidebar for configuration options
    with st.sidebar:
        st.subheader("Configuration Options")

        # Preprocessing settings
        st.subheader("Preprocessing Settings")
        apply_equalization = st.checkbox("Apply Histogram Equalization", value=True)
        blur_kernel_size = st.slider("Blur Kernel Size", min_value=1, max_value=25, value=5, step=2)

        # Pectoral removal settings
        st.subheader("Pectoral Muscle Removal Settings")
        side_selection = st.selectbox("Select Side", ["Left", "Right"])  # Add side selection
        start_x = st.slider('Start X', 0, 1024, 100)
        start_y = st.slider('Start Y', 0, 1024, 200)
        end_x = st.slider('End X', 0, 1024, 800)
        end_y = st.slider('End Y', 0, 1024, 900)

        # Display settings
        st.subheader("Display Settings")
        display_original = st.checkbox("Show Original Image", value=True)
        display_preprocessed = st.checkbox("Show Preprocessed Image", value=True)
        display_cleaned = st.checkbox("Show Cleaned Image", value=True)
        display_thresholded = st.checkbox("Show Thresholded Image", value=True)
        display_dense_region = st.checkbox("Show Dense Region Image", value=True)

    # Image upload (supports drag-and-drop)
    uploaded_file = st.file_uploader("Upload a mammogram image (Drag and Drop or Click to Upload)", type=["png", "jpg", "jpeg", "dcm"])
    if uploaded_file is not None:
        try:
            # Check if the file is a DICOM file
            if uploaded_file.name.endswith(".dcm"):
                image = load_dicom(uploaded_file)
            else:
                # Handle non-DICOM files (PNG, JPG, JPEG)
                image = Image.open(uploaded_file)
                image = np.array(image.convert('L'))

            # Apply preprocessing based on user choices
            preprocessed_image = image.copy()
            if apply_equalization:
                preprocessed_image = cv2.equalizeHist(preprocessed_image)
            preprocessed_image = cv2.GaussianBlur(preprocessed_image, (blur_kernel_size, blur_kernel_size), 0)

            # Adjustments begin after the preprocessed image
            cleaned_image = remove_pectoral_muscle(preprocessed_image.copy(), side_selection, (start_x, start_y), (end_x, end_y))

            # Detect the most dense region
            thresholded_image, dense_image = find_highest_dense_region(cleaned_image)

            # Display images based on user selections
            st.subheader("Processed Images")
            col1, col2 = st.columns(2)

            if display_original:
                with col1:
                    st.image(image, caption="Original Image", use_column_width=True)
            
            if display_preprocessed:
                with col2:
                    st.image(preprocessed_image, caption="Preprocessed Image", use_column_width=True)
            
            col3, col4 = st.columns(2)

            if display_cleaned:
                with col3:
                    st.image(cleaned_image, caption="After Pectoral Removal", use_column_width=True)
            
            if display_thresholded:
                with col4:
                    st.image(thresholded_image, caption="Thresholded Image", use_column_width=True)

            # Display the dense region image with reduced size
            if display_dense_region:
                st.subheader("Dense Region")
                st.image(dense_image, caption="Highest Dense Region (Red)", use_column_width=False, width=400)  # Set width to 400

        except Exception as e:
            st.error(f"An error occurred: {e}")

   # Chatbot implementation
st.subheader("Ask EchoBot")  # Tag for the chatbot section

# Add a medical-themed image for the chatbot
st.image("data:image/jpeg;base64,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", width=50)  # Use your medical-themed image

# Create a text input for user input with a specific key for state management
user_input = st.text_input("You: ➡️", "", key="chat_input")  # Use a key for managing input state

if user_input:
    message(user_input, is_user=True)  # Display user message
    # Get a response from the AI model
    response = query_aixplain_model(user_input)
    message(response)  # Display bot response

    # Note: No need to clear the input; Streamlit manages this automatically with the key.


if __name__ == "__main__":
    main()





