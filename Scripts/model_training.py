import Models.model as m


def train_model(train_loader):
    for epoch in range(m.num_epochs):

        for i, (inputs, targets) in enumerate(m.train_loader):
            # Forward pass
            outputs = m.model(inputs.float())
            targets = targets.long()
            loss = m.criterion(outputs, targets)

            # Backward pass and optimization
            m.optimizer.zero_grad()
            loss.backward()
            m.optimizer.step()

            if (i+1) % 100 == 0:
                print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch+1, m.num_epochs, i+1, len(m.train_loader), loss.item()))

