"""
Predicts the class of a given image using a TinyVGG model
"""
import torch
import torchvision
import argparse
import model_builder
 
parser = argparse.ArgumentParser()
parser.add_argument("--image", help="Target image to predict the class of")
parser.add_argument("--model_path", default="models/05_pytorch_going_modular_tinyvgg_args.pth", 
                        type=str, help="Target model to use for the class prediction")

args = parser.parse_args()

IMG_PATH = args.image
MODEL_PATH = args.model_path
class_names = ["pizza", "steak", "sushi"]

def load_model(path=MODEL_PATH):
    model = model_builder.TinyVGG(
        in_shape=3,
        hidden=10,
        out_shape=3
    )
    model.load_state_dict(torch.load(path))
    return model

def predict_on_image(image_path=IMG_PATH, model_path=MODEL_PATH):
    model = load_model(MODEL_PATH)
    image = torchvision.io.read_image(str(image_path)).type(torch.float32)
    image = image / 255.
    transform = torchvision.transforms.Resize(size=(64, 64))
    image = transform(image)
    
    model.eval()
    with torch.inference_mode():
        pred_logits = model(image.unsqueeze(dim=0))
        pred_probs = torch.softmax(pred_logits, dim=1)
        pred_label = torch.argmax(pred_probs, dim=1)
        pred_label_class = class_names[pred_label]
        
    print(f"[INFO] Pred class: {pred_label_class}, Pred prob: {pred_probs.max():.3f}")
    
if __name__ == "__main__":
    predict_on_image()
