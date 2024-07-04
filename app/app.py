import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import io

# Supongamos que tienes un modelo de PyTorch preentrenado cargado
# Aquí se carga un modelo ficticio
class SimpleModel(torch.nn.Module):
    def forward(self, x):
        # Aquí se define una operación de ejemplo: invertir los colores de la imagen
        return 1 - x

model = SimpleModel()

# Definir transformaciones para preprocesar la imagen
preprocess = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

# Función para procesar la imagen con el modelo
def process_image(image):
    image_tensor = preprocess(image).unsqueeze(0)  # Añadir una dimensión para el batch
    with torch.no_grad():
        output_tensor = model(image_tensor)
    output_image = transforms.ToPILImage()(output_tensor.squeeze())  # Quitar la dimensión del batch y convertir a PIL
    return output_image

# Interfaz de Streamlit
st.title("Image IRL to invincible cartoon")
uploaded_file = st.file_uploader("Sube una imagen", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Imagen original')

    processed_image = process_image(image)
    st.image(processed_image, caption='Imagen procesada')
