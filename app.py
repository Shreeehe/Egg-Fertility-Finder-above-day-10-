import streamlit as st
from ultralytics import YOLO
import tempfile
from PIL import Image

# This sets up the page for our app
st.title("Fertile and Infertile Egg Classifier")
st.write("Upload an egg image to see if it is fertile or not.")

# This loads our trained model
# Make sure the 'best.pt' file is in the same folder as this app.py file!
model = YOLO('best.pt')

# This creates a button to upload a picture
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# This part runs when a picture is uploaded
if uploaded_file is not None:
    # We show the picture in the app
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    # Here, we use our model to make a prediction
    # We save the picture for a moment so the model can read it
    results = model.predict(source=image)
    # This is how we look at what the model said
    if results:
        for result in results:
            names_dict = result.names
            class_indices = result.boxes.cls.tolist()
            if class_indices:
                predicted_class = names_dict[int(class_indices[0])]
                if predicted_class == 'Fertile':
                    st.write("Alert! Our model has confirmed this egg is hosting a tiny party inside, so it's Fertile! Best not to crash it.")
                else:
                    st.write("This egg's life motto is 'No plans, just vibes.' Our model has classified it as Infertile.")
            else:
                    st.write("My model is asking, 'Is this a potato? A rock? A very round... pillow?' It has no idea! No egg found.")


