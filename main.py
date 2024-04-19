import os
import json
from PIL import Image

import numpy as np
import tensorflow as tf
import streamlit as st

# Load the pre-trained model
working_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(working_dir, "trained_model", "plant_disease_prediction_model.h5")
model = tf.keras.models.load_model(model_path)

# Load class indices
class_indices_path = os.path.join(working_dir, "class_indices.json")
with open(class_indices_path, "r") as f:
    class_indices = json.load(f)

# Define plant care instructions and disease reasons
plant_care_and_diseases = {
    'apple': {
    'care_instructions': "Apple trees require full sunlight and well-drained soil. Water them regularly, especially during dry periods, and fertilize them in early spring. Prune the trees annually to remove dead or diseased branches and encourage airflow. Watch out for pests like apple maggots and codling moths.",
    'disease_reasons': "Apple trees can be affected by diseases such as apple scab, powdery mildew, fire blight, and cedar apple rust. These diseases are often caused by fungi or bacteria and can spread through water, air, or infected plant material. Overcrowding, excessive humidity, and poor pruning practices can also contribute to disease development."
},
'blueberry': {
    'care_instructions': "Blueberry bushes thrive in acidic, well-drained soil and require full sunlight. Keep the soil consistently moist, especially during fruit development. Prune the bushes annually to remove dead or overcrowded branches and promote new growth. Protect the bushes from birds with netting or reflective tape.",
    'disease_reasons': "Blueberry bushes are susceptible to diseases such as powdery mildew, mummy berry, and root rot. These diseases can be caused by fungi or bacteria and are often spread through water, air, or contaminated soil. Overwatering, poor drainage, and overcrowding can also contribute to disease outbreaks."
},
'cherry': {
    'care_instructions': "Cherry trees need full sunlight and well-drained soil. Water them deeply and regularly, especially during dry spells, and fertilize them in early spring. Prune the trees annually to remove dead or diseased branches and improve airflow. Protect the fruit from birds with netting or scare devices.",
    'disease_reasons': "Cherry trees can be affected by diseases such as brown rot, cherry leaf spot, powdery mildew, and bacterial canker. These diseases are often caused by fungi or bacteria and can spread through water, air, or infected plant material. Overcrowding, excessive humidity, and poor sanitation practices can also contribute to disease outbreaks."
},
'grape': {
    'care_instructions': "Grapevines require full sunlight and well-drained soil. Water them regularly, especially during dry periods, and fertilize them in early spring. Prune the vines annually to remove dead or diseased wood and promote fruit production. Protect the grapes from birds with netting or bird scare devices.",
    'disease_reasons': "Grapevines are susceptible to diseases such as powdery mildew, downy mildew, black rot, and botrytis bunch rot. These diseases are often caused by fungi and can spread through water, air, or infected plant material. Overcrowding, excessive humidity, and poor airflow can also contribute to disease development."
},
'orange': {
    'care_instructions': "Orange trees need full sunlight and well-drained soil. Water them deeply and regularly, especially during dry periods, and fertilize them in spring and summer. Prune the trees annually to remove dead or diseased branches and improve airflow. Protect the trees from frost in cooler climates.",
    'disease_reasons': "Orange trees can be affected by diseases such as citrus canker, citrus greening, and root rot. These diseases are often caused by bacteria or fungi and can spread through water, air, or infected plant material. Overwatering, poor drainage, and nutrient deficiencies can also contribute to disease development."
},
'peach': {
    'care_instructions': "Peach trees require full sunlight and well-drained soil. Water them deeply and regularly, especially during dry periods, and fertilize them in early spring. Prune the trees annually to remove dead or diseased branches and improve airflow. Protect the fruit from pests like peach borers and plum curculios.",
    'disease_reasons': "Peach trees are susceptible to diseases such as peach leaf curl, brown rot, powdery mildew, and bacterial spot. These diseases are often caused by fungi or bacteria and can spread through water, air, or infected plant material. Overcrowding, excessive humidity, and poor sanitation practices can also contribute to disease outbreaks."
},
'pepper': {
    'care_instructions': "Pepper plants need full sunlight and well-drained soil. Water them regularly, especially during dry periods, and fertilize them with a balanced fertilizer. Mulch around the plants to retain moisture and suppress weeds. Watch out for pests like aphids and pepper maggots.",
    'disease_reasons': "Pepper plants can be affected by diseases such as bacterial leaf spot, powdery mildew, and phytophthora blight. These diseases are often caused by bacteria, fungi, or oomycetes and can spread through water, air, or contaminated soil. Overwatering, poor drainage, and overcrowding can also contribute to disease development."
},
'potato': {
    'care_instructions': "Potato plants require well-drained, loose soil and full sunlight. Plant them in trenches or mounds and keep the soil evenly moist. Hill up soil around the stems as they grow to prevent tubers from greening. Watch out for pests like potato beetles and aphids.",
    'disease_reasons': "Potato plants are susceptible to diseases such as late blight, early blight, potato scab, and blackleg. These diseases are often caused by fungi or bacteria and can spread through contaminated soil, water, or infected seed potatoes. Overcrowding, poor crop rotation, and improper storage can also contribute to disease outbreaks."
},
'raspberry': {
    'care_instructions': "Raspberry bushes need full sunlight and well-drained soil. Water them regularly, especially during dry periods, and mulch around the plants to retain moisture and suppress weeds. Prune the canes annually to remove old or diseased growth and encourage new shoots. Protect the plants from pests like raspberry beetles and spider mites.",
    'disease_reasons': "Raspberry bushes can be affected by diseases such as raspberry leaf spot, anthracnose, powdery mildew, and cane blight. These diseases are often caused by fungi or bacteria and can spread through water, air, or infected plant material. Overcrowding, excessive humidity, and poor sanitation practices can also contribute to disease outbreaks."
},
'soybean': {
    'care_instructions': "Soybeans need full sunlight and well-drained soil. Plant them in rows with good spacing to promote airflow and reduce disease risk. Water them regularly, especially during flowering and pod development. Watch out for pests like soybean aphids and bean leaf beetles.",
    'disease_reasons': "Soybeans can be affected by diseases such as soybean rust, sudden death syndrome, white mold, and brown stem rot. These diseases are often caused by fungi and can spread through contaminated soil, infected seed, or windborne spores. Overcrowding, poor drainage, and high humidity can also contribute to disease development."
},
'squash': {
    'care_instructions': "Squash plants need full sunlight and well-drained soil. Plant them in hills or mounds with good spacing to promote airflow and reduce disease risk. Water them regularly, especially during fruit development, and mulch around the plants to retain moisture and suppress weeds. Watch out for pests like squash bugs and vine borers.",
    'disease_reasons': "Squash plants can be affected by diseases such as powdery mildew, downy mildew, bacterial wilt, and cucumber mosaic virus. These diseases are often caused by fungi, bacteria, or viruses and can spread through contaminated soil, water, or infected plant debris. Overcrowding, poor air circulation, and inconsistent watering can also contribute to disease development."
},
'strawberry': {
    'care_instructions': "Strawberry plants require full sunlight and well-drained soil. Plant them in raised beds or rows with good spacing to promote airflow and reduce disease risk. Water them regularly, especially during fruit development, and mulch around the plants to retain moisture and suppress weeds. Protect the fruit from pests like slugs and birds.",
    'disease_reasons': "Strawberry plants can be affected by diseases such as powdery mildew, gray mold, anthracnose, and verticillium wilt. These diseases are often caused by fungi or bacteria and can spread through contaminated soil, water, or infected plant material. Overcrowding, poor drainage, and lack of proper sanitation can also contribute to disease outbreaks."
},
'tomato': {
    'care_instructions': "Tomato plants require regular watering, ample sunlight, and support for their vines. They are susceptible to diseases like blight and need to be monitored regularly.",
    'disease_reasons': "Tomato plants can be affected by various diseases such as early blight, late blight, powdery mildew, and tomato mosaic virus. These diseases are often spread through contaminated soil, water, or infected plant debris. Overcrowding, poor air circulation, and inconsistent watering can also contribute to disease development."
}

    # Add more plant care instructions and disease reasons for other plants as needed
}

# Function to Load and Preprocess the Image using Pillow
def load_and_preprocess_image(image_path, target_size=(224, 224)):
    img = Image.open(image_path)
    img = img.resize(target_size)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array.astype('float32') / 255.
    return img_array

# Function to Predict the Class of an Image
def predict_image_class(model, image_path, class_indices):
    preprocessed_img = load_and_preprocess_image(image_path)
    predictions = model.predict(preprocessed_img)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class_name = class_indices[str(predicted_class_index)]
    return predicted_class_name

# Streamlit App
st.set_page_config(page_title="PlantAI: Your Plant Companion", page_icon="🌱")

st.title('PlantAI: Your Plant Companion')

st.sidebar.title('Menu')
selected_option = st.sidebar.selectbox('Select an option', ['Plant Disease Detection', 'Plant Care Advice'])

if selected_option == 'Plant Disease Detection':
    st.subheader('Detect Plant Disease')
    uploaded_image = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        if st.button('Detect Disease'):
            prediction = predict_image_class(model, uploaded_image, class_indices)
            st.success(f'Prediction: {prediction}')

elif selected_option == 'Plant Care Advice':
    st.subheader('Plant Care Advice')
    selected_plant = st.selectbox('Select a plant', list(plant_care_and_diseases.keys()))

    care_instructions = plant_care_and_diseases[selected_plant]['care_instructions']
    disease_reasons = plant_care_and_diseases[selected_plant]['disease_reasons']

    st.write(f"**Care Instructions for {selected_plant.capitalize()} Plants:**")
    st.write(care_instructions)

    st.write(f"**Common Diseases and Reasons for {selected_plant.capitalize()} Plants:**")
    st.write(disease_reasons)
