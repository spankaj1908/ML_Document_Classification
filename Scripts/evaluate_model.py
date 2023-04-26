import Models.model as m
import torch

# Let's evaluate the performance of the trained model on the testing set
m.model.eval()
with torch.no_grad():
    correct_pred_count = 0
    total = 0
    #first_run = True
    for inputs, targets in m.test_loader:
        #if first_run:
            #print("Type of input is:")
            #print(type(inputs))
            #first_run =  False
        outputs = m.model(inputs.float())
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct_pred_count += (predicted == targets).sum().item()

    accuracy = (correct_pred_count / total)*100
    print('Accuracy of the network on the test set: {:.2f}%'.format(accuracy))