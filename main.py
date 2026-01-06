from torch import nn
import torch
import matplotlib.pyplot as plt

# Create the class
class linearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Weights and bias parameters create
        self.weights = nn.Parameter(torch.randn(1,
                                                 requires_grad=True,
                                                 dtype = torch.float))
        
        self.bias = nn.Parameter(torch.randn(1,
                                              requires_grad=True,
                                              dtype = torch.float))
    # Overwrite forward function    
    def forward(self, x :torch.Tensor) -> torch.Tensor:
        return self.weights * x + self.bias
    
# Set the seed
torch.manual_seed(42)

model_0 = linearRegressionModel()

# Create weights and bias
weight = 0.6
bias = 0.2

start = 0
end = 1
step = 0.01

X = torch.arange(start, end, step).unsqueeze(dim=1)

# Linear Regression Formula
y = weight * X + bias


# Create train, test split

train_split = int(0.8 * len(X))
X_train, y_train = X[:train_split], y[:train_split]
X_test, y_test = X[train_split:], y[train_split:]

# Plotting predictions
def plot_predictions_of_model(train_data = X_train,
                              train_label = y_train,
                              test_data = X_test,
                              test_labels = y_test,
                              predictions = None):
    
    plt.scatter(train_data, train_label, s = 4, label = "Training Data")

    plt.scatter(test_data, test_labels, c = "g", s= 4, label = "Test Data")

    if predictions is not None:
        plt.scatter(test_data, predictions, c = "r", s = 4, label = "Prediction")

    plt.legend()

with torch.inference_mode():
    y_prediction = model_0(X_test)
    y_predictions = y_prediction[:10]


loss_function = nn.MSELoss()

optimizer = torch.optim.ASGD(params = model_0.parameters(),
                             lr = 0.01)



torch.manual_seed(42)

epochs = 100

for epoch in range(epochs):
    model_0.train()

    optimizer.zero_grad()                
    y_pred = model_0(X_train)
    loss = loss_function(y_pred, y_train)

    loss.backward()
    optimizer.step()

    model_0.eval()

    with torch.inference_mode():

        test_pred = model_0(X_test)

        test_loss = loss_function(test_pred, y_test)

    if epoch % 10 == 0:
        print(
            f"Epoch: {epoch} |"
            f"Train Loss: {loss.item():.6f} |"
            f"Test Loss: {test_loss.item():.6f}"
        )
        print(model_0.state_dict())

with torch.inference_mode():
    y_pred_new = model_0(X_test)


pred = plot_predictions_of_model(predictions = y_pred_new)
plt.show(block = True)

