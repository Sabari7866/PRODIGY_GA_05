# Neural Style Transfer - Circle with Gradient (Final Balanced Version)

!pip install torch torchvision matplotlib pillow --quiet

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from torchvision.models import VGG19_Weights
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import requests
from io import BytesIO

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

def load_image(img, max_size=256, shape=None):
    if isinstance(img, str):
        image = Image.open(img).convert('RGB')
    else:
        image = img.convert('RGB')

    size = max_size if max(image.size) > max_size else max(image.size)
    if shape is not None:
        size = shape

    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    image = transform(image)[:3, :, :].unsqueeze(0)
    return image.to(device)

def create_circle_gradient(img_size=256):
    img = Image.new("RGB", (img_size, img_size), color="white")
    draw = ImageDraw.Draw(img)

    for y in range(img_size):
        color_val = int(255 * y / img_size)
        draw.line([(0, y), (img_size, y)], fill=(color_val, 0, 255 - color_val))

    mask = Image.new("L", (img_size, img_size), 0)
    mask_draw = ImageDraw.Draw(mask)
    mask_draw.ellipse([30, 30, img_size-30, img_size-30], fill=255)
    img.putalpha(mask)
    img = img.convert("RGB")
    return img

content_img = create_circle_gradient()

style_url = "https://pytorch.org/tutorials/_static/img/neural-style/picasso.jpg"
response = requests.get(style_url)
style_img = Image.open(BytesIO(response.content)).convert("RGB")

content = load_image(content_img)
style = load_image(style_img, shape=[content.size(2), content.size(3)])

class VGGFeatures(nn.Module):
    def __init__(self):
        super().__init__()
        self.vgg = models.vgg19(weights=VGG19_Weights.DEFAULT).features.to(device).eval()
        self.content_layers = ['21']
        self.style_layers = ['0', '5', '10', '19', '28']

    def forward(self, x):
        content_features = {}
        style_features = {}
        for name, layer in self.vgg._modules.items():
            x = layer(x)
            if name in self.content_layers:
                content_features[name] = x
            if name in self.style_layers:
                style_features[name] = x
        return content_features, style_features

def gram_matrix(tensor):
    b, c, h, w = tensor.size()
    tensor = tensor.view(c, h * w)
    gram = torch.mm(tensor, tensor.t())
    return gram / (c * h * w)

vgg = VGGFeatures().to(device).eval()

target = torch.randn_like(content).requires_grad_(True).to(device)


optimizer = optim.Adam([target], lr=0.003)

style_weights = {'0': 1.0, '5': 0.8, '10': 0.5, '19': 0.3, '28': 0.1}

content_weight = 1e3
style_weight = 150

print("Starting style transfer...")

for step in range(300):

    optimizer.zero_grad()

    content_features_t, style_features_t = vgg(target)
    content_features_c, _ = vgg(content)
    _, style_features_s = vgg(style)

    content_loss = torch.mean((content_features_t['21'] - content_features_c['21'])**2)

    style_loss = 0
    for layer in style_weights:
        target_feature = style_features_t[layer]
        style_feature = style_features_s[layer]

        gm_target = gram_matrix(target_feature)
        gm_style = gram_matrix(style_feature)

        style_loss += style_weights[layer] * torch.mean((gm_target - gm_style)**2)

    total_loss = content_weight * content_loss + style_weight * style_loss
    total_loss.backward()
    optimizer.step()

    with torch.no_grad():
        target.clamp_(-3, 3)

    if step % 50 == 0:
        print(f"Step {step}: Total Loss: {total_loss.item():.4f}")

print("Style transfer complete!")

def im_convert(tensor):
    image = tensor.to("cpu").clone().detach()
    image = image.squeeze(0)
    image = image * torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
    image = image + torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
    image = image.clamp(0,1)
    image = transforms.ToPILImage()(image)
    return image

plt.figure(figsize=(15,5))

plt.subplot(1,3,1)
plt.title("Content Image")
plt.imshow(content_img)
plt.axis('off')

plt.subplot(1,3,2)
plt.title("Style Image")
plt.imshow(style_img)
plt.axis('off')

plt.subplot(1,3,3)
plt.title("Output (Stylized)")
plt.imshow(im_convert(target))
plt.axis('off')

plt.show()
