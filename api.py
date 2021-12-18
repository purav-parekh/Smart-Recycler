import os
import torch
import io
from flask import Flask, render_template, request
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
import pickle

app = Flask(__name__)
FOLDER = 'C:/Users/purav parekh/Desktop/MSCS/Sem 1/Intro_to_CS/Project/frontend/'

DEVICE = torch.device('cpu')

DATASET = ['non-recyclable', 'recyclable']


def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))


class ImageClassificationBase(nn.Module):
    def training_step(self, batch):
        images, labels = batch
        out = self(images)                  # Generate predictions
        loss = F.cross_entropy(out, labels)  # Calculate loss
        return loss


class ResNet(ImageClassificationBase):
    def __init__(self):
        super().__init__()
        self.network = models.resnet50(
            pretrained=True)  # Use a pretrained model
        num_ftrs = self.network.fc.in_features  # Replace last layer
        self.network.fc = nn.Linear(num_ftrs, len(DATASET))

    def forward(self, xb):
        return torch.sigmoid(self.network(xb))


class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
            return super().find_class(module, name)


"""Move tensor(s) to chosen device"""


def to_device(data, device):
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)


with open(os.path.join(FOLDER, 'trained_model.pkl'), 'rb') as f:
    MODEL = CPU_Unpickler(f).load()


def predict_image(img, model):
    xb = to_device(img.unsqueeze(0), DEVICE)  # Convert to a batch of 1
    yb = model(xb)  # Get predictions from model
    prods, preds = torch.max(yb, dim=1)  # Pick index with highest probability
    return DATASET[preds[0].item()]  # Retrieve the class label


def predict_external_image(image_name, model):
    transformations = transforms.Compose(
        [transforms.Resize((256, 256)), transforms.ToTensor()])
    image = Image.open(image_name)
    example_image = transformations(image)
    plt.imshow(example_image.permute(1, 2, 0))
    val = predict_image(example_image, model)
    return val


@app.route('/', methods=["GET", 'POST'])
def upload_predict():
    if request.method == "POST":
        image_file = request.files["image"]
        if image_file:
            image_loc = os.path.join(
                FOLDER + 'static/img/', image_file.filename)
            image_file.save(image_loc)
            pred = predict_external_image(image_loc, MODEL)
            if pred == 'recyclable':
                sentence = 'Good Job! You used an object which is '
                sentence2 = 'Thank You!'
            else:
                sentence = 'Uh oh! You used an object which is '
                sentence2 = 'Please use recyclable objects!'
            return render_template("index.html", prediction=sentence + pred + '! ' + sentence2, img='static/img/'+image_file.filename)
        else:
            return render_template("index.html", prediction='Please upload an image.')
    return render_template("index.html", prediction="Please select an image.")


if __name__ == "__main__":
    app.run(port=1200, debug=True)
