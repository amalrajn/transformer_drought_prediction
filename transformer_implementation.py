import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

class TransformerModel(nn.Module):
    def __init__(self, input_dim, model_dim, num_head, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.transformer = nn.Transformer(d_model = model_dim, num_head=num_head, num_encoder_layers=num_encoder_layers, 
                                          num_decoder_layers=num_decoder_layers,dim_feedforward=dim_feedforward, dropout=dropout)
        self.fc_in = nn.Linear(input_dim, model_dim)
        self.fc_out = nn.Linear(model_dim,1)
        self.positional_encoding = nn.Parameter(torch.zeros(1,1000, model_dim)) 
    
    def forward(self, src):
        src = self.fc_in(src) + self.positional_encoding[:, :src.size(1), :]
        transformer_output = self.transformer(src, src)
        output = self.fc_out(transformer_output[:,-1,:])
        return output
    
    def train_model(model, train_loader, val_loader, num_epochs = 50, learning_rate = 0.001):
        criterion = nn.MSEloss()
        optimizer = optim.Adam(model.parameters(), lr = learning_rate)

        for epoch in range(num_epochs):
            model.train()
            train_loss = 0
            for inputs, targets in train_loader:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            val_loss = 0
            model.eval()
            with torch.no_grad():
                for inputs, targets in val_loader:
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    val_loss += loss.item()
            
            print(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss / len(train_loader)}, Val Loss: {val_loss / len(val_loader)}')

    def evaluate_model(model, test_loader):
        model.eval()
        all_targets = []
        all_predictions = []
        with torch.no_grad():
            for inputs, targets in test_loader:
                outputs = model(inputs)
                all_targets.append(targets.numpy())
                all_predictions.append(outputs.numpy())
        all_targets = np.concatenate(all_targets)
        all_predictions = np.concatenate(all_predictions)

        mse = np.mean((all_targets - all_predictions) ** 2)
        print(f'Test MSE: {mse}')

        f2 = fbeta_score(all_targets, np.round(all_predictions), beta=2)
        precision = precision_score(all_targets, np.round(all_predictions))
        recall = recall_score(all_targets, np.round(all_predictions))
        print(f'F2 Score: {f2}, Precision: {precision}, Recall: {recall}')

        cm = confusion_matrix(all_targets, np.round(all_predictions))
        print(f'Confusion Matrix:\n{cm}')

        plt.figure(figsize=(10,5))
        plt.subplot(1, 2, 1)
        plt.plot(all_targets, label='Actual')
        plt.plot(all_predictions, label='Predicted')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(range(1, len(all_predictions) + 1), all_predictions, label='Predicted')
        plt.plot(range(1, len(all_targets) + 1), all_targets, label='Actual')
        plt.xlabel('Samples')
        plt.ylabel('Values')
        plt.legend()
        plt.show()
