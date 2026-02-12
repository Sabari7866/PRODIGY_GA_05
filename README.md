# PRODIGY_GA_05
🚀  Task-05 Completed | Prodigy InfoTech | Generative AI Internship
🎨 Neural Style Transfer – Circle with Gradient

📌 Project Overview

This project demonstrates Neural Style Transfer (NST) using a pre-trained VGG19 convolutional neural network in PyTorch.

The goal of this task was to:

Apply the artistic style of a famous painting to a custom-generated content image (Circle with Gradient).

Instead of using a regular photograph as content, a synthetic gradient circle image was programmatically generated and stylized using a classical painting.

🖼️ Example Output
Content Image	Style Image	Stylized Output
Gradient Circle	Picasso Painting	Stylized Circle

The final result preserves the structure of the circle while blending the artistic texture of the painting.

🧠 Concepts Used

Convolutional Neural Networks (CNN)

Transfer Learning

VGG19 Pre-trained Model

Content Loss

Style Loss

Gram Matrix for Style Representation

Optimization using Adam

Image Normalization & De-normalization

⚙️ How It Works

Content Image Creation

A gradient-filled circle is generated using PIL.

Style Image Loading

A famous painting is used as the style reference.

Feature Extraction

VGG19 extracts:

Content features from deeper layers

Style features from multiple convolutional layers

Loss Calculation

Content Loss → Preserves structure

Style Loss → Transfers artistic texture using Gram matrices

Optimization

The target image is optimized using backpropagation until the total loss is minimized.

🛠️ Technologies Used

Python

PyTorch

Torchvision

Matplotlib

PIL (Python Imaging Library)

🚀 Installation & Execution
1️⃣ Clone the repository
git clone https://github.com/your-username/neural-style-transfer.git
cd neural-style-transfer

2️⃣ Install dependencies
pip install torch torchvision matplotlib pillow

3️⃣ Run the script
python style_transfer.py


Or run in Jupyter/Colab.

📊 Key Parameters
content_weight = 1e3
style_weight = 150
optimizer = Adam(lr=0.003)
image_size = 256


These values were tuned to balance content preservation and artistic stylization.

🔍 Challenges Faced

Handling NaN loss values due to exploding gradients

Balancing content and style weights

Stylizing smooth synthetic gradients (limited structure)

Optimizing performance on CPU vs GPU

🎯 Learning Outcomes

Deep understanding of feature representations in CNNs

Practical implementation of style transfer

Understanding Gram matrices in texture modeling

Handling optimization stability issues

Working with CUDA acceleration

📌 Internship Details

Organization: Prodigy InfoTech
Domain: Generative AI
Task: Task-05 – Neural Style Transfer


📷 Future Improvements

Experiment with different style images

Use VGG16 for faster performance

Implement multi-style blending

Convert into a web application
#OUTPUT


<img width="816" height="377" alt="image" src="https://github.com/user-attachments/assets/5e49852a-e04d-4a23-bd14-648202aaaa0d" />
