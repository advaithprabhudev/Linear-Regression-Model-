# Linear-Regression-Model

This model is a Linear Regression Model that uses the formula `y = a + bX`, where a is the **bias**, and a the **weights**. The following Python script uses PyTorch t do the machine learning behind the scenes. Also uses matplotlib to plot the predictions against the test and training data.

Download the following libraries using the following line:

`pip install matplotlib torch`

This model is not completely accurate, due to the fatal error in the Linear Regression Formula.

For the loss function, I use the MSE (Mean Squared Error), you can use any error as you fit by changing the:

`loss_function = nn.MSELoss()` 

For the optimizer, you can change the learning rate and optimizer as needed in the line:

`optimizer = torch.optim.ASGD(params = model_0.parameters(),
                              lr = 0.01)`

Thank you for seeing this small project of mine!

<img width="547" height="413" alt="image" src="https://github.com/user-attachments/assets/574aeadd-4792-4619-9dbc-f8f3f8347ba8" />
