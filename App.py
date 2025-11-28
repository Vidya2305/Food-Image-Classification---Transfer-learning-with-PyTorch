# 8. Deployment (optional) - Export model with TorchScript or create a Streamlit app for inference.

import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import gradio as gr

device = "cuda" if torch.cuda.is_available() else "cpu"

model = models.resnet50(weights=None)

num_features = model.fc.in_features
model.fc = torch.nn.Linear(num_features, 10)

state_dict = torch.load("best_model_auto_selected.pth", map_location=device)
model.load_state_dict(state_dict)

model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

classes = [
    "cannoli",
    "ceviche",
    "crab_cakes",
    "frozen_yogurt",
    "gnocchi",
    "grilled_cheese_sandwich",
    "onion_rings",
    "pork_chop",
    "ravioli",
    "spaghetti_bolognese"
]

def predict(image):
    image = Image.fromarray(image)
    img_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]

    result = {classes[i]: float(probabilities[i]) for i in range(10)}

    return result

with gr.Blocks() as demo:
    gr.Markdown(
        """
        <h1 style='text-align:center; color:#4A90E2;'>🍽️ Food Image Classifier</h1>
        <p style='text-align:center; font-size:16px;'>Upload a food image and the model will predict the top class.</p>
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            with gr.Accordion("ℹ️ About This App", open=False):
                gr.Markdown(
                    """
                    This is a **Deep Learning model-based Food Image Classifier** built using:
                    - **PyTorch** for model inference  
                    - **Gradio** for UI  
                    - **Transfer Learning (ResNet/EfficientNet)**  

                    ### **How It Works**
                    1. You upload a food image  
                    2. The model preprocesses the image (resize + normalize)  
                    3. It outputs **probabilities for 10 food classes**  
                    4. Top 1 prediction is displayed    
                    """
                )

        with gr.Column(scale=2):
            image_input = gr.Image(type="numpy", label="📸 Upload Food Image")

            output_label = gr.Label(
                num_top_classes=1,
                label="🍔 Prediction Probability"
            )

            gr.Interface(
                fn=predict,
                inputs=image_input,
                outputs=output_label,
            )


demo.launch(share=True)
