# Technical Approach: Smart Product Pricing Challenge

## 1. Executive Summary
Our approach to the **Smart Product Pricing Challenge** is centered around a **multimodal deep learning model** designed to holistically analyze both the textual and visual information provided for each product.  

By combining state-of-the-art techniques from **Natural Language Processing (NLP)** and **Computer Vision (CV)**, our end-to-end model learns to map the rich features from product descriptions and images directly to their market price.  

The core of our methodology is the **fusion of pre-trained Transformer and Convolutional Neural Network (CNN)** models to create a comprehensive understanding of each product.

---

## 2. Data Preprocessing & Feature Engineering

To prepare the data for our model, we implemented several key preprocessing and feature engineering steps:

###  Text Data (`catalog_content`)
- The raw text content for each product is processed using a **pre-trained DistilBERT tokenizer**.  
- This converts the text into a format suitable for a Transformer model, capturing the semantic meaning of the product's title and description.

###  Image Data (`image_link`)
- Product images are downloaded from their respective URLs.  
- Each image is resized to a standard **224×224** dimension and normalized using standard **ImageNet statistics**.  
- For any products with missing or broken image links, we substitute a **blank placeholder image** to ensure the training pipeline remains robust and does not crash.

###  Target Variable Transformation (`price`)
- Product prices typically have a **right-skewed distribution**.  
- To create a more stable training target and better align with the **SMAPE** evaluation metric, we apply a **logarithmic transformation**:

  \[
  \text{target} = \log(1 + \text{price})
  \]

- All model predictions are transformed back to the original scale for final submission:

  \[
  \text{price} = e^{\text{prediction}} - 1
  \]

---

## 3. Model Architecture

Our solution is a **two-branch, end-to-end neural network** that processes text and images in parallel before fusing them for a final prediction.

###  Text Branch (NLP)
- Uses a **pre-trained DistilBERT** model.  
- The tokenized `catalog_content` is passed through DistilBERT.  
- The final hidden state of the special `[CLS]` token is extracted as a **768-dimensional contextual embedding** representing the textual features of the product.

###  Image Branch (Computer Vision)
- Employs a **pre-trained EfficientNet-B0** as the vision backbone.  
- The final classification layer is removed, and the model outputs a **1280-dimensional feature vector** summarizing the key visual attributes.

###  Fusion & Regression Head
- The text and image embeddings are **concatenated** to form a single combined representation.  
- This fused vector is passed through a **Multi-Layer Perceptron (MLP)** head:
  - Two dense layers with **ReLU activation**  
  - **Dropout** for regularization  
  - A final single neuron outputs the predicted **log-price** of the product.

---

## 4. Training & Evaluation Strategy

###  Loss Function
- **Mean Squared Error (MSE)** between predicted and true log-prices.

###  Optimizer
- **AdamW**, which is well-suited for fine-tuning Transformer-based architectures.

###  Evaluation
- The model is validated using **Symmetric Mean Absolute Percentage Error (SMAPE)** on a **10% validation split** of the training data.  
- The model checkpoint with the **lowest validation SMAPE** is saved for final inference on the test set.

---

