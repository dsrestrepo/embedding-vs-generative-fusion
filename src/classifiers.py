import os

import pandas as pd
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import MultiLabelBinarizer

# Function to train classic ML model
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
import copy


# Metrics
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import RocCurveDisplay

import time

import warnings
# Suppress all warnings
warnings.filterwarnings("ignore")

from src.data_utils import split_data, process_labels

######### Models and Evaluation #########

# Early Fusion Model
class EarlyFusionModel(nn.Module):
    """
    Early Fusion Model for combining text and image features.

    Args:
    - text_input_size (int): Dimension of the text input.
    - image_input_size (int): Dimension of the image input.
    - output_size (int): Dimension of the output.
    - hidden (int or list): Hidden layer(s) size(s) for the model.

    Attributes:
    - fc1 (nn.Sequential): First fully connected layer(s).
    - fc2 (nn.Linear): Second fully connected layer.

    Methods:
    - forward(text, image): Forward pass of the model.

    Example:
    model = EarlyFusionModel(text_input_size=512, image_input_size=256, output_size=10, hidden=[128, 64])
    """
    def __init__(self, text_input_size, image_input_size, output_size, hidden=[128], p=0.2):
        super(EarlyFusionModel, self).__init__()
        
        output_dim = text_input_size + image_input_size
        self.p = p
        
        # Initialize layers as an empty list
        layers = []
        
        # Add the linear layer and ReLU activation if 'hidden' is an integer
        if isinstance(hidden, int):
            if hidden > 0:
                layers.append(nn.Linear(output_dim, hidden))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(p=self.p))
                output_dim = hidden
            # If hidden is 0 or less, we add nothing (linear probing logic handled by default output_dim)
            
        # Add the linear layer and ReLU activation for each element in 'hidden' if it's a list
        elif isinstance(hidden, list):
            for h in hidden:
                layers.append(nn.Linear(output_dim, h))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(p=self.p))
                layers.append(nn.BatchNorm1d(h))
                output_dim = h
        
        # If no layers were added (linear Probe), this sequential is effectively Identity if empty? 
        # Actually nn.Sequential with no args is identity.
        self.fc1 = nn.Sequential(*layers)

        #self.fc1 = nn.Linear(text_input_size + image_input_size, hidden)
        
        self.fc2 = nn.Linear(output_dim, output_size)

    def forward(self, text, image):
        x = torch.cat((text, image), dim=1)
        #x = torch.relu(self.fc1(x))
        x = self.fc1(x)
        x = self.fc2(x)
        return x

# Late Fusion Model
class LateFusionModel(nn.Module):
    """
    Late Fusion Model for combining text and image features.

    Args:
    - text_input_size (int): Dimension of the text input.
    - image_input_size (int): Dimension of the image input.
    - output_size (int): Dimension of the output.
    - hidden_images (int or list): Hidden layer(s) size(s) for the image features.
    - hidden_text (int or list): Hidden layer(s) size(s) for the text features.

    Attributes:
    - text_fc (nn.Sequential): Fully connected layers for text features.
    - image_fc (nn.Sequential): Fully connected layers for image features.
    - fc2 (nn.Linear): Second fully connected layer.

    Methods:
    - forward(text, image): Forward pass of the model.

    Example:
    model = LateFusionModel(text_input_size=512, image_input_size=256, output_size=10, hidden_images=[64], hidden_text=[64])
    """
    def __init__(self, text_input_size, image_input_size, output_size, hidden_images=[64], hidden_text=[64], p=0.2):
        super(LateFusionModel, self).__init__()
        
        self.p = p
        
        self.text_fc, out_text = self._get_layers(text_input_size, hidden_text, p=self.p)
        self.image_fc, out_images = self._get_layers(image_input_size, hidden_images, p=self.p)
        
        #self.text_fc = nn.Linear(text_input_size, hidden_text)
        #self.image_fc = nn.Linear(image_input_size, hidden_images)
        
        
        self.fc2 = nn.Linear(out_text + out_images, output_size)
        
    def _get_layers(self, embed_dim, hidden, p=0.2):
        # Initialize layers as an empty list
        layers = []
        output_dim = embed_dim
        
        # Add the linear layer and ReLU activation if 'hidden' is an integer
        if isinstance(hidden, int):
            if hidden > 0:
                layers.append(nn.Linear(output_dim, hidden))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(p=p))
                output_dim = hidden
            # If hidden is 0, no layers added
            
        # Add the linear layer and ReLU activation for each element in 'hidden' if it's a list
        elif isinstance(hidden, list):
            for h in hidden:
                layers.append(nn.Linear(output_dim, h))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(p=p))
                layers.append(nn.BatchNorm1d(h))
                output_dim = h
        
        fc = nn.Sequential(*layers)
        
        return fc, output_dim

    def forward(self, text, image):
        text_output = self.text_fc(text)
        image_output = self.image_fc(image)
        #text_output = torch.relu(self.text_fc(text))
        #image_output = torch.relu(self.image_fc(image))
        x = torch.cat((text_output, image_output), dim=1)
        x = self.fc2(x)
        return x



def test_model(y_test, y_pred, V=True):
    """
    Evaluates the model on the training and test data respectively
    1. Predictions on test data
    2. Classification report
    3. Confusion matrix
    4. ROC curve

    Inputs:
    y_test: numpy array with test labels
    y_pred: numpy array with predicted test labels
    """
    
    plot_matrix = False
    if y_pred.shape[1] < 102:
        plot_matrix = True
        
    
    if y_pred.shape[1] > 1:
        y_test = np.argmax(y_test, axis=1)
        y_pred = np.argmax(y_pred, axis=1)
    
    if V:
        # Confusion matrix
        # Create a confusion matrix of the test predictions
        if plot_matrix:
            cm = confusion_matrix(y_test, y_pred)
            # create heatmap
            # Set the size of the plot
            fig, ax = plt.subplots(figsize=(15, 15))
            sns.heatmap(cm, annot=True, cmap='Blues', fmt='g', ax=ax)
            # Set plot labels
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.title('Confusion Matrix')
            # Display plot
            plt.show()

        #create ROC curve
        from sklearn.preprocessing import LabelBinarizer
        fig, ax = plt.subplots(figsize=(15, 15))

        label_binarizer = LabelBinarizer().fit(y_test)
        y_onehot_test = label_binarizer.transform(y_test)
        y_onehot_pred = label_binarizer.transform(y_pred)

        if (y_onehot_pred.shape[1] < 2):
            fpr, tpr, _ = roc_curve(y_test,  y_pred)

            #create ROC curve
            #plt.plot(fpr,tpr)
            RocCurveDisplay.from_predictions(
                    y_test,
                    y_pred,
                    name=f"ROC curve",
                    color='aqua',
                    ax=ax,
                )
            plt.plot([0, 1], [0, 1], "k--", label="ROC curve for chance level (AUC = 0.5)")
            plt.title('ROC Curve')
            plt.ylabel('True Positive Rate')
            plt.xlabel('False Positive Rate')
            plt.show()

        else:
            from itertools import cycle
            colors = cycle(["aqua", "darkorange", "cornflowerblue", "red", "green", "yellow", "purple", "pink", "brown", "black"])

            for class_id, color in zip(range(len(label_binarizer.classes_)), colors):
                RocCurveDisplay.from_predictions(
                    y_onehot_test[:, class_id],
                    y_onehot_pred[:, class_id],
                    name=f"ROC curve for {label_binarizer.classes_[class_id]}",
                    color=color,
                    ax=ax,
                )

            plt.plot([0, 1], [0, 1], "k--", label="ROC curve for chance level (AUC = 0.5)")
            plt.axis("square")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title("Extension of Receiver Operating Characteristic\nto One-vs-Rest multiclass")
            plt.legend(loc='upper left', bbox_to_anchor=(1, 1), bbox_transform=plt.gcf().transFigure)
            plt.show()

        # Classification report
        # Create a classification report of the test predictions
        cr = classification_report(y_test, y_pred)
        # print classification report
        print(cr)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')  # Use weighted average for multi-class precision
    recall = recall_score(y_test, y_pred, average='weighted')  # Use weighted average for multi-class recall
    f1 = f1_score(y_test, y_pred, average='weighted')  # Use weighted average for multi-class F1-score

    return accuracy, precision, recall, f1

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
def train_early_fusion(train_loader, test_loader, text_input_size, image_input_size, output_size, num_epochs=5, multilabel=True, report=False, lr=3e-4, set_weights=True, adam=False, p=0.0, V=True, val_loader=None, patience=5, hidden=[128]):
    """
    Train an Early Fusion Model.

    Args:
    - train_loader (DataLoader): DataLoader for the training set.
    - test_loader (DataLoader): DataLoader for the testing set.
    - text_input_size (int): Dimension of the text input.
    - image_input_size (int): Dimension of the image input.
    - output_size (int): Dimension of the output.
    - num_epochs (int): Number of training epochs.
    - multilabel (bool): Flag for multilabel classification.
    - report (bool): Flag to generate a classification report, confusion matrix, and ROC curve.
    - hidden (int or list): Hidden layer configuration.

    Example:
    train_early_fusion(train_loader, test_loader, text_input_size=512, image_input_size=256, output_size=10, num_epochs=5, multilabel=True)
    """
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True

    model = EarlyFusionModel(text_input_size=text_input_size, image_input_size=image_input_size, output_size=output_size, p=p, hidden=hidden)
    model = model.to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    
    
    
    print(f'The number of parameters of the model are: {count_parameters(model)}')
    
    if set_weights:
        # Convert tensor to numpy for sklearn compatibility if needed or use torch operations
        labels_tensor = train_loader.dataset.labels
        
        if not multilabel:
            # Check if labels are 1D (binary/sparse) or 2D (one-hot)
            if labels_tensor.ndim == 1:
                class_indices = labels_tensor.cpu().numpy().astype(int)
            else:
                class_indices = torch.argmax(labels_tensor, dim=1).cpu().numpy()

            # Compute class weights using class indices
            class_weights = compute_class_weight('balanced', classes=np.unique(class_indices), y=class_indices)
            class_weights = torch.tensor(class_weights, dtype=torch.float32)
        else:
            class_counts = labels_tensor.sum(dim=0)
            total_samples = len(labels_tensor)
            num_classes = labels_tensor.shape[1]
            
            # Use torch operations for class weights calculation
            # Add small epsilon or manipulate to avoid div by zero if necessary, though balanced usually assumes existence
            class_weights = total_samples / (num_classes * class_counts)

            # Convert class_weights to a PyTorch tensor
            class_weights = class_weights.float()
    else:
        class_weights = None
        
    if class_weights is not None:
        class_weights = class_weights.to(device)

    if multilabel:
        criterion = nn.BCEWithLogitsLoss(weight=class_weights)
    elif(output_size == 1):
        # For binary classification with single output, we use pos_weight
        if class_weights is not None and len(class_weights) == 2:
            pos_weight = class_weights[1] / class_weights[0]
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        else:
            criterion = nn.BCEWithLogitsLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    if adam:
        optimizer = optim.Adam(model.parameters(), lr=lr)
    else:
        optimizer = optim.AdamW(model.parameters(), lr=lr)

    # Early stopping variables
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_wts = copy.deepcopy(model.state_dict())
    best_epoch = 0

    for epoch in range(num_epochs):
        
        model.train()

        for batch in train_loader:
            text, image, labels = batch['text'].to(device, non_blocking=True), batch['image'].to(device, non_blocking=True), batch['labels'].to(device, non_blocking=True)
            optimizer.zero_grad()
            outputs = model(text, image)
            
            if output_size == 1 and labels.ndim == 1:
                labels = labels.unsqueeze(1)
                
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        # Validation for early stopping
        if val_loader:
             model.eval()
             val_loss = 0.0
             with torch.no_grad():
                 for batch in val_loader:
                     text, image, labels = batch['text'].to(device), batch['image'].to(device), batch['labels'].to(device)
                     outputs = model(text, image)
                     
                     if output_size == 1 and labels.ndim == 1:
                        labels = labels.unsqueeze(1)
                        
                     loss = criterion(outputs, labels)
                     val_loss += loss.item() * text.size(0)
             
             val_loss = val_loss / len(val_loader.dataset)
             print(f"Epoch {epoch + 1}/{num_epochs} - Validation Loss: {val_loss:.4f}")
             
             if val_loss < best_val_loss:
                 best_val_loss = val_loss
                 best_model_wts = copy.deepcopy(model.state_dict())
                 epochs_no_improve = 0
                 best_epoch = epoch + 1
             else:
                 epochs_no_improve += 1
                 if epochs_no_improve >= patience:
                     print(f"Early stopping triggered after {epoch + 1} epochs")
                     break
        else:
             print(f"Epoch {epoch + 1}/{num_epochs}")
             # Save last weights if no validation
             best_model_wts = copy.deepcopy(model.state_dict())
             best_epoch = epoch + 1

    # Load best model weights
    if best_model_wts:
        model.load_state_dict(best_model_wts)

    # Final evaluation on test set
    model.eval()
    with torch.no_grad():
        y_true, y_pred = [], []
        for batch in test_loader:
            text, image, labels = batch['text'].to(device), batch['image'].to(device), batch['labels'].to(device)
            outputs = model(text, image)
            if multilabel or (output_size == 1):
                preds = torch.sigmoid(outputs)
            else:
                preds = torch.softmax(outputs, dim=1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

        y_true, y_pred = np.array(y_true), np.array(y_pred)
                    
        if multilabel or (output_size == 1):
            y_pred_one_hot = (y_pred > 0.5).astype(int)
        else:
            predicted_class_indices = np.argmax(y_pred, axis=1)
            # Convert the predicted class indices to one-hot encoding
            y_pred_one_hot = np.eye(y_pred.shape[1])[predicted_class_indices]
            
        test_accuracy = accuracy_score(y_true, y_pred_one_hot)
        f1 = f1_score(y_true, y_pred_one_hot, average='macro')
        
        # Compute AUC for binary or multi-class classification
        if multilabel or (output_size == 1):
            if output_size == 1:  # Binary classification case
                auc_scores = roc_auc_score(y_true, y_pred, average=None)
            else:  # Multi-label classification
                try:
                    auc_scores = roc_auc_score(y_true, y_pred, average=None, multi_class='ovr')
                except ValueError:
                    auc_scores = np.nan
        else:  # Multi-class classification
            try:
                auc_scores = roc_auc_score(y_true, y_pred_one_hot, average=None, multi_class='ovr')
            except ValueError:
                auc_scores = np.nan
        
        if output_size == 1:
            macro_auc = auc_scores
        else:    
            macro_auc = np.nanmean(auc_scores)

        if V:
            print(f"Final Test Evaluation (Best Epoch {best_epoch}) - Accuracy: {test_accuracy:.4f}, Macro-F1: {f1:.4f}, Macro-AUC: {macro_auc:.4f}")

    # Construct best dictionary
    best = {
        'Acc': {'Acc': test_accuracy, 'F1': f1, 'Auc': macro_auc, 'Epoch': best_epoch, 'Auc_Per_Class': auc_scores},
        'Macro-F1': {'Acc': test_accuracy, 'F1': f1, 'Auc': macro_auc, 'Epoch': best_epoch, 'Auc_Per_Class': auc_scores},
        'AUC': {'Acc': test_accuracy, 'F1': f1, 'Auc': macro_auc, 'Epoch': best_epoch, 'Auc_Per_Class': auc_scores}
    }
            
    if report:
        accuracy, precision, recall, f1 = test_model(y_true, y_pred_one_hot, V)
        return accuracy, precision, recall, f1, best
    else:
        return test_accuracy, None, None, f1, best

            
            

# Function to train late fusion model (similar changes)
def train_late_fusion(train_loader, test_loader, text_input_size, image_input_size, output_size, num_epochs=5, multilabel=True, report=False, lr=3e-4, set_weights=True, adam=False, p=0.0, V=True, val_loader=None, patience=5, hidden=[128]): 
    """
    Train a Late Fusion Model.

    Args:
    - train_loader (DataLoader): DataLoader for the training set.
    - test_loader (DataLoader): DataLoader for the testing set.
    - text_input_size (int): Dimension of the text input.
    - image_input_size (int): Dimension of the image input.
    - output_size (int): Dimension of the output.
    - num_epochs (int): Number of training epochs.
    - multilabel (bool): Flag for multilabel classification.
    - report (bool): Flag to generate a classification report, confusion matrix, and ROC curve.
    - hidden (int or list): Hidden layer configuration.

    Example:
    train_late_fusion(train_loader, test_loader, text_input_size=512, image_input_size=256, output_size=10, num_epochs=5, multilabel=True)
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
        
    model = LateFusionModel(text_input_size=text_input_size, image_input_size=image_input_size, output_size=output_size, p=p, hidden_images=hidden, hidden_text=hidden)
    
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    
    model.to(device)
    
    print(f'The number of parameters of the model are: {count_parameters(model)}')
    
    if set_weights:
        # Convert tensor to numpy for sklearn compatibility if needed or use torch operations
        labels_tensor = train_loader.dataset.labels

        if not multilabel:
            # Check if labels are 1D (binary/sparse) or 2D (one-hot)
            if labels_tensor.ndim == 1:
                class_indices = labels_tensor.cpu().numpy().astype(int)
            else:
                class_indices = torch.argmax(labels_tensor, dim=1).cpu().numpy()

            # Compute class weights using class indices
            class_weights = compute_class_weight('balanced', classes=np.unique(class_indices), y=class_indices)
            class_weights = torch.tensor(class_weights, dtype=torch.float32)
        else:
            class_counts = labels_tensor.sum(dim=0)
            total_samples = len(labels_tensor)
            num_classes = labels_tensor.shape[1]
            
            # Use torch operations for class weights calculation
            class_weights = total_samples / (num_classes * class_counts)

            # Convert class_weights to a PyTorch tensor
            class_weights = class_weights.float()
    else:
        class_weights = None
        
    if class_weights is not None:
        class_weights = class_weights.to(device)

    if multilabel:
        criterion = nn.BCEWithLogitsLoss(weight=class_weights)
    elif(output_size == 1):
        # For binary classification with single output, we use pos_weight
        if class_weights is not None and len(class_weights) == 2:
            pos_weight = class_weights[1] / class_weights[0]
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        else:
            criterion = nn.BCEWithLogitsLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        
    
    #print(f'The number of parameters of the model are: {count_parameters(model)}')
    
    #if multilabel or (output_size == 1):
    #    criterion = nn.BCEWithLogitsLoss()
    #else:
    #    criterion = nn.CrossEntropyLoss()
    
    if adam:
        optimizer = optim.Adam(model.parameters(), lr=lr)
    else:
        optimizer = optim.AdamW(model.parameters(), lr=lr)
    
       # Early stopping variables
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_wts = copy.deepcopy(model.state_dict())
    best_epoch = 0

    for epoch in range(num_epochs):

        model.train()
        for batch in train_loader:
            text, image, labels = batch['text'].to(device, non_blocking=True), batch['image'].to(device, non_blocking=True), batch['labels'].to(device, non_blocking=True)
            optimizer.zero_grad()
            outputs = model(text, image)
            
            if output_size == 1 and labels.ndim == 1:
                labels = labels.unsqueeze(1)
            
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
                        
        # Validation for early stopping
        if val_loader:
             model.eval()
             val_loss = 0.0
             with torch.no_grad():
                 for batch in val_loader:
                     text, image, labels = batch['text'].to(device), batch['image'].to(device), batch['labels'].to(device)
                     outputs = model(text, image)
                     
                     if output_size == 1 and labels.ndim == 1:
                        labels = labels.unsqueeze(1)
                        
                     loss = criterion(outputs, labels)
                     val_loss += loss.item() * text.size(0)
             
             val_loss = val_loss / len(val_loader.dataset)
             print(f"Epoch {epoch + 1}/{num_epochs} - Validation Loss: {val_loss:.4f}")
             
             if val_loss < best_val_loss:
                 best_val_loss = val_loss
                 best_model_wts = copy.deepcopy(model.state_dict())
                 epochs_no_improve = 0
                 best_epoch = epoch + 1
             else:
                 epochs_no_improve += 1
                 if epochs_no_improve >= patience:
                     print(f"Early stopping triggered after {epoch + 1} epochs")
                     break
        else:
             print(f"Epoch {epoch + 1}/{num_epochs}")
             # Save last weights if no validation
             best_model_wts = copy.deepcopy(model.state_dict())
             best_epoch = epoch + 1

    # Load best model weights
    if best_model_wts:
        model.load_state_dict(best_model_wts)

    # Final evaluation on test set
    model.eval()
    with torch.no_grad():
        y_true, y_pred = [], []
        for batch in test_loader:
            text, image, labels = batch['text'].to(device), batch['image'].to(device), batch['labels'].to(device)
            outputs = model(text, image)
            if multilabel or (output_size == 1):
                preds = torch.sigmoid(outputs)
            else:
                preds = torch.softmax(outputs, dim=1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

        y_true, y_pred = np.array(y_true), np.array(y_pred)
        
        if multilabel or (output_size == 1):
            y_pred_one_hot = (y_pred > 0.5).astype(int)
        else:
            predicted_class_indices = np.argmax(y_pred, axis=1)
            # Convert the predicted class indices to one-hot encoding
            y_pred_one_hot = np.eye(y_pred.shape[1])[predicted_class_indices]
            
        test_accuracy = accuracy_score(y_true, y_pred_one_hot)
        f1 = f1_score(y_true, y_pred_one_hot, average='macro')
        
        # Compute AUC for binary or multi-class classification
        if multilabel or (output_size == 1):
            if output_size == 1:  # Binary classification case
                auc_scores = roc_auc_score(y_true, y_pred, average=None)
            else:  # Multi-label classification
                try:
                    auc_scores = roc_auc_score(y_true, y_pred, average=None, multi_class='ovr')
                except ValueError:
                    auc_scores = np.nan
        else:  # Multi-class classification
            try:
                auc_scores = roc_auc_score(y_true, y_pred_one_hot, average=None, multi_class='ovr')
            except ValueError:
                auc_scores = np.nan
        
        if output_size == 1:
            macro_auc = auc_scores
        else:    
            macro_auc = np.nanmean(auc_scores)

        if V:
            print(f"Final Test Evaluation (Best Epoch {best_epoch}) - Accuracy: {test_accuracy:.4f}, Macro-F1: {f1:.4f}, Macro-AUC: {macro_auc:.4f}")

    # Construct best dictionary
    best = {
        'Acc': {'Acc': test_accuracy, 'F1': f1, 'Auc': macro_auc, 'Epoch': best_epoch, 'Auc_Per_Class': auc_scores},
        'Macro-F1': {'Acc': test_accuracy, 'F1': f1, 'Auc': macro_auc, 'Epoch': best_epoch, 'Auc_Per_Class': auc_scores},
        'AUC': {'Acc': test_accuracy, 'F1': f1, 'Auc': macro_auc, 'Epoch': best_epoch, 'Auc_Per_Class': auc_scores}
    }
            
    if report:
        accuracy, precision, recall, f1 = test_model(y_true, y_pred_one_hot, V)
        return accuracy, precision, recall, f1, best
    else:
        return test_accuracy, None, None, f1, best
            
# Function to evaluate classic ML model
def evaluate_classic_ml_model(model_name, y_true, y_pred, train_columns):
    """
    Evaluate the performance of classic ML models.

    Args:
    - model_name (str): Name of the ML model.
    - y_true (np.ndarray): True labels.
    - y_pred (np.ndarray): Predicted labels.
    - train_columns (list): List of columns from the training set.

    Example:
    evaluate_classic_ml_model("Random Forest", y_test, rf_pred, train_columns)
    """
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='micro')

    print(f"{model_name} - Test Accuracy: {accuracy}")
    print(f"{model_name} - Test F1 Score: {f1}")
    
def train_classic_ml_models(train_data, test_data, train_labels, test_labels, train_columns):
    """
    Train and evaluate classic ML models.

    Args:
    - train_data (pd.DataFrame): DataFrame containing training data.
    - test_data (pd.DataFrame): DataFrame containing testing data.
    - train_labels (np.ndarray): Labels for the training set.
    - test_labels (np.ndarray): Labels for the testing set.
    - train_columns (list): List of columns from the training set.

    Example:
    train_classic_ml_models(train_data, test_data, train_labels, test_labels, train_columns)
    """
    # Separate features and labels
    X_train, y_train = train_data[text_columns + image_columns], train_labels
    X_test, y_test = test_data[text_columns + image_columns], test_labels

    # Random Forest
    rf_model = OneVsRestClassifier(RandomForestClassifier())
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)

    # Logistic Regression
    lr_model = OneVsRestClassifier(LogisticRegression())
    lr_model.fit(X_train, y_train)
    lr_pred = lr_model.predict(X_test)

    # SVM
    svm_model = OneVsRestClassifier(SVC())
    svm_model.fit(X_train, y_train)
    svm_pred = svm_model.predict(X_test)

    # Evaluate models
    evaluate_classic_ml_model("Random Forest", y_test, rf_pred, train_columns)
    evaluate_classic_ml_model("Logistic Regression", y_test, lr_pred, train_columns)
    evaluate_classic_ml_model("SVM", y_test, svm_pred, train_columns)