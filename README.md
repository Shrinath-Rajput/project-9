# project-9
classify chili powder and adultered chili powder

🌶️ Chili Powder Adulteration Detection using CNN (MobileNetV2, VGG16, ResNet50)
📌 Project Description
This project aims to classify pure and adulterated red chili powder using deep learning techniques. Adulteration in spices, especially with substances like brick powder, poses serious health hazards. The goal is to build a reliable image classification system to detect such adulteration from image data using Convolutional Neural Networks (CNNs) with transfer learning models.

🧠 Models Used
MobileNetV2

VGG16

ResNet50

These pre-trained models are fine-tuned for our binary classification task:

Pure Chili Powder

Adulterated Chili Powder (e.g., mixed with brick powder)

📂 Dataset
The dataset contains two classes: pure/ and adulterated/.

Around 3000+ images were used.
Data split: 70% training / 30% testing
All images were resized to 224x224 before training.
Note: Dataset is not publicly shared here due to size/licensing. You can prepare your own dataset or contact for access.

🛠️ Tools & Libraries
Python
TensorFlow / Keras
OpenCV (for image preprocessing)
NumPy, Pandas
Scikit-learn (for evaluation metrics)
Matplotlib / Seaborn (for plotting graphs)

⚙️ Project Workflow
Data Preprocessing
Load and label images
Resize and normalize images
Split into training/testing sets
Model Building
Use transfer learning with MobileNetV2, VGG16, ResNet50
Fine-tune final layers for binary classification
Apply dropout to reduce overfitting
Model Training
Use categorical_crossentropy or binary_crossentropy
Fit on training set and validate on test set
Track accuracy and loss per epoch
Evaluation
Accuracy, Precision, Recall, F1-Score
Confusion Matrix
ROC Curve
Model Comparison
Compare the performance of all three models
elect the best-performing one

📊 Results
Model	Accuracy	Precision	Recall	F1-Score
MobileNetV2	95%	96%	94%	95%
VGG16	93%	92%	94%	93%
ResNet50	96%	97%	95%	96%

These results may vary depending on the dataset and hardware.

📈 Sample Outputs
✅ Correctly classified pure/adulterated images
📉 Accuracy & loss curves
🔁 Confusion matrix
📍 Visual comparison between models
🧪 Future Enhancements
Add more adulterants (e.g., salt powder, talcum)
Use real-time webcam detection
Deploy model using Flask or Streamlit

🧑‍💻 Author
Shrinath Rajput
