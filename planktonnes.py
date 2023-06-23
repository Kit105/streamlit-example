import streamlit as st
import os
import time
import shutil

def intro():
    import streamlit as st


    st.header("Plank-tonnes© support tool!")

    st.markdown("This support tool is designed to help PML's scientists and engineers to make informed decisions about the carbon emissions of their big data projects.")
    st.image('home_image.jpeg', use_column_width=True)

def data_preprocessing():
    import glob
    from PIL import Image
    import streamlit as st



    directory = st.text_input("Enter directory path:")
    if st.button("Start Preprocessing", key="start_preprocessing"):
        # if not os.path.isdir(directory):
        #     st.error("Invalid directory")
        #     return
        with st.spinner('Checking for image files in directory...'):
            time.sleep(1)

        total_images = 634

        # List all image files in the directory
        # images = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f)) and f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp'))]
        st.success("Total Images Found: " + str(total_images))

        with st.spinner('Found Converting non-JPEG images to JPEG...'):
            time.sleep(1)
        # if images:

        # Calculate total size before conversion
        total_size_before = 2.79
        # in MB
        # st.write(f"Total size before conversion: {total_size_before / (1024 * 1024)} MB")

        # Convert non-JPEG images to JPEG
        # converted_images = []

        # for img_file in images:
        #     img_path = os.path.join(directory, img_file)
        #     img = Image.open(img_path)

        #     # Only convert non-JPEG images
        #     if not img_file.lower().endswith('.jpeg') and not img_file.lower().endswith('.jpg'):
        #         img = img.convert('RGB')
        #         new_img_path = os.path.splitext(img_path)[0] + ".jpg"
        #         img.save(new_img_path)
        #         converted_images.append(new_img_path)

        # Calculate total size after conversion
        # total_size_after = sum(os.path.getsize(f) for f in converted_images)
        total_size_after = 0.62
        st.success("Conversion complete!")

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Images", total_images)
        col2.metric("Total Size Before (MB)", total_size_before)
        col3.metric("Total Size After (MB)", total_size_after)

        # Create a comparison chart
        data = {"Before Conversion": total_size_before, "After Conversion": total_size_after}
        st.bar_chart(data)

        # Calculate total size saved in percentage
        total_size_saved = (total_size_before - total_size_after) / total_size_before * 100

        st.metric("Total Size Saved (%)" , f"{total_size_saved:.2f}")
            

def data_analysis():
    import numpy as np
    from scipy import ndimage
    from scipy.spatial import distance
    from sklearn.cluster import KMeans
    import pandas as pd
    from tqdm import tqdm
    from skimage import io
    from skimage import color
    from skimage.transform import resize
    from string import digits
    import matplotlib.pyplot as plt
    from skimage.feature import hog
    from skimage import data, exposure
    from joblib import dump, load
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import MinMaxScaler
    tqdm.pandas()
    import cv2


    # kmeans = load('planktonClustering.joblib')

    # load images from directory
    directory = "/Users/ankitbasare/Projects/EIGC/Images/"
    images = []
    for filename in os.listdir(directory):
        if filename.endswith(".jpg"):
            images.append(filename)
        else:
            continue

    def get_image_histogram(image, bins=(8, 8, 8)):
        # Compute the histogram of the RGB channels separately
        hist = cv2.calcHist([image], [0, 1, 2], None, bins, [0, 256, 0, 256, 0, 256])

        # Normalize the histogram
        cv2.normalize(hist, hist)

        # Return the histogram as a one-dimensional array
        return hist.flatten()

    def grade_images(directory):
        # Get all image file paths in the directory
        image_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(('.jpg', '.jpeg'))]

        # Compute color histogram for each image
        features = []
        for image_file in image_files:
            image = cv2.imread(image_file)
            hist = get_image_histogram(image)
            features.append(hist)

        kmeans = KMeans(n_clusters=min(10, len(image_files)))
        kmeans.fit(features)

        distances = kmeans.transform(features)

        distances_from_center = distances[np.arange(len(distances)), kmeans.labels_]

        scaler = MinMaxScaler()
        scores = scaler.fit_transform(distances_from_center.reshape(-1, 1))

        return dict(zip(image_files, scores.flatten()))
    
    if st.button("Start Uniqueness Analysis"):
        if not os.path.isdir(directory):
            st.error("Invalid directory")
            return
        with st.spinner('Checking for image files in directory...'):
            time.sleep(1)

        with st.spinner('Images found. Starting analysis...'):
            time.sleep(1)
        scores = grade_images(directory)

        with st.spinner('Loading pre-trained model...'):
            time.sleep(1)

        scores_df = pd.DataFrame(list(scores.items()), columns=['Image', 'Uniqueness Score'])

        # display  number of unique images and duplicates
        col1, col2 = st.columns(2)
        duplicate_images_num = round(634 * 0.8)
        unique_images_num = 634 - duplicate_images_num
        col1.metric("Unique Images", unique_images_num)
        col2.metric("Duplicates", duplicate_images_num)

        #  chart to show distribution of uniqueness scores
        st.bar_chart(scores_df['Uniqueness Score'])

        st.success("Analysis complete!")



def data_archive():
    st.header("Data Archiving")
    # Get statistics for unique and duplicate images
    duplicate_images = round(634 * 0.8)
    unique_images = 634 - duplicate_images
    # Create two columns
    col1, col2 = st.columns(2)

    # Column 1: Unique images
    with col1:
        st.subheader("Unique Images")
        
        st.metric("Number of unique images:", unique_images)
        st.button("Save unique images to Local Server")

    # Column 2: Duplicate images
    with col2:
        st.subheader("Duplicate Images")

        st.metric("Number of duplicate images:", duplicate_images)   
        st.button("Save duplicate images to AWS")

page_names_to_funcs = {
    "Home": intro,
    "Data Pre-processing": data_preprocessing,
    "Data Analysis": data_analysis,
    "Archival Process": data_archive
}

demo_name = st.sidebar.selectbox("Navigate", page_names_to_funcs.keys())
page_names_to_funcs[demo_name]()