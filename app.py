Python 3.13.5 (tags/v3.13.5:6cb20a2, Jun 11 2025, 16:15:46) [MSC v.1943 64 bit (AMD64)] on win32
Enter "help" below or click "Help" above for more information.
import streamlit as st
from ultralytics import YOLO
... import tempfile
... from PIL import Image
... 
... # This sets up the page for our app
... st.title("Fertile and Infertile Egg Classifier")
... st.write("Upload an egg image to see if it is fertile or not.")
... 
... # This loads our trained model
... # Make sure the 'best.pt' file is in the same folder as this app.py file!
... model = YOLO('best.pt')
... 
... # This creates a button to upload a picture
... uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
... 
... # This part runs when a picture is uploaded
... if uploaded_file is not None:
...     # We show the picture in the app
...     image = Image.open(uploaded_file)
...     st.image(image, caption='Uploaded Image', use_column_width=True)
...     st.write("")
...     st.write("Classifying...")
... 
...     # Here, we use our model to make a prediction
...     # We save the picture for a moment so the model can read it
...     with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
...         tmp_file.write(uploaded_file.read())
...         results = model.predict(source=tmp_file.name)
... 
...     # This is how we look at what the model said
...     if results:
...         for result in results:
...             names_dict = result.names
...             class_indices = result.boxes.cls.tolist()
...             if class_indices:
...                 predicted_class = names_dict[int(class_indices[0])]
...                 st.write(f"The model thinks this egg is: **{predicted_class}** 💖")
...             else:
