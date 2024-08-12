import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import fbeta_score, confusion_matrix, recall_score, precision_score
import time
import matplotlib.pyplot as plt

class TransformerModel(nn.Module):
    def __init__(self, input_dim, model_dim, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout):
        super(TransformerModel, self).__init__()
        self.model_dim = model_dim
        self.fc_in = nn.Linear(input_dim, model_dim)
        self.positional_encoding = nn.Parameter(torch.zeros(1, 120, model_dim))
        self.transformer = nn.Transformer(d_model=model_dim, nhead=nhead, num_encoder_layers=num_encoder_layers, 
                                          num_decoder_layers=num_decoder_layers, dim_feedforward=dim_feedforward, dropout=dropout)
        self.fc_out = nn.Linear(model_dim, input_dim)

    def forward(self, src):
        if src.dim() == 2:
            src = src.unsqueeze(1)
        elif src.dim() == 3:
            src = src.permute(1, 0, 2) 
        else:
            raise ValueError(f"Unexpected input dimensions {src.shape}")

        src = self.fc_in(src) + self.positional_encoding[:, :src.size(0), :]
        transformer_output = self.transformer(src, src)
        output = self.fc_out(transformer_output[-1])
        return output

def format_time(seconds):
    hrs = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hrs} hours, {mins} minutes, {secs} seconds"

def train_model(model, train_loader, val_loader, num_epochs, learning_rate, device):
    start_time = time.time()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    train_losses = []
    val_losses = []
    train_metrics = {}
    val_metrics = {}
    
    cumulative_examples = 0
    train_f2_scores = []
    val_f2_scores = []
    example_counts = []

    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        model.train()
        running_loss = 0.0
        all_train_targets = []
        all_train_outputs = []
        
        for inputs, _ in train_loader:
            inputs = inputs.to(device)
            inputs = inputs.unsqueeze(1)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, inputs.squeeze(1))
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            
            all_train_targets.extend(inputs.cpu().numpy())
            all_train_outputs.extend(outputs.cpu().detach().numpy())
            
            # Track number of examples processed
            cumulative_examples += inputs.size(0)
            example_counts.append(cumulative_examples)
            
            # Calculate F2 score for training data
            train_metrics = calculate_metrics(all_train_targets, all_train_outputs)
            train_f2_scores.append(train_metrics['f2_score'])
        
        train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(train_loss)
        
        model.eval()
        val_loss = 0.0
        all_val_targets = []
        all_val_outputs = []
        with torch.no_grad():
            for inputs, _ in val_loader:
                inputs = inputs.to(device)
                inputs = inputs.unsqueeze(1)
                outputs = model(inputs)
                loss = criterion(outputs, inputs.squeeze(1))
                val_loss += loss.item() * inputs.size(0)
                
                all_val_targets.extend(inputs.cpu().numpy())
                all_val_outputs.extend(outputs.cpu().detach().numpy())
        
        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)
        
        # Calculate F2 score for validation data
        val_metrics = calculate_metrics(all_val_targets, all_val_outputs)
        val_f2_scores.append(val_metrics['f2_score'])
        
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        epoch_end_time = time.time()
        epoch_duration = format_time(epoch_end_time - epoch_start_time)
        print(f"Epoch {epoch+1} training time: {epoch_duration}")
        print(f"Train F2 Score: {train_metrics['f2_score']:.4f}, Val F2 Score: {val_metrics['f2_score']:.4f}")

    end_time = time.time()
    print('The training process took:', format_time(end_time - start_time)) 

    # Plot the training and validation F2 score curves
    plt.figure(figsize=(10, 5))
    plt.plot(example_counts, train_f2_scores, label='Training F2 Score', color='r')
    #plt.plot(example_counts, val_f2_scores, label='Validation F2 Score', color='g')
    plt.xlabel('Number of Examples')
    plt.ylabel('F2 Score')
    plt.title('Training F2 Score Curve')
    plt.legend()
    plt.grid(True)
    plt.show()

    return train_losses, val_losses, train_metrics, val_metrics
def evaluate_model(model, test_loader, device):
    model.eval()
    test_loss = 0.0
    criterion = nn.MSELoss()
    all_targets = []
    all_outputs = []
    with torch.no_grad():
        for inputs, _ in test_loader:
            inputs = inputs.to(device)  # Move inputs to device
            inputs = inputs.unsqueeze(1)
            outputs = model(inputs)
            loss = criterion(outputs, inputs.squeeze(1))
            test_loss += loss.item() * inputs.size(0)
            
            all_targets.extend(inputs.cpu().numpy())
            all_outputs.extend(outputs.cpu().detach().numpy())
    
    test_loss /= len(test_loader.dataset)
    test_metrics = calculate_metrics(all_targets, all_outputs)
    
    return test_loss, test_metrics

def calculate_metrics(targets, outputs):
    targets = (np.array(targets) > 0.5).astype(int).flatten()
    outputs = (np.array(outputs) > 0.5).astype(int).flatten()
    
    f2 = fbeta_score(targets, outputs, beta=2)
    recall = recall_score(targets, outputs)
    precision = precision_score(targets, outputs)
    conf_matrix = confusion_matrix(targets, outputs)
    
    return {
        'f2_score': f2,
        'recall': recall,
        'precision': precision,
        'confusion_matrix': conf_matrix
    }