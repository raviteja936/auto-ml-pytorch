import torch
from src.evaluate.evalloop import EvalLoop


class TrainLoop:
    def __init__(self, model, train_loader, optimizer, loss_fn, val_loader=None, print_every=100):
        self.model = model
        self.model.train()
        self.train_loader = train_loader
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.print_every = print_every
        if val_loader:
            self.eval = EvalLoop(model, val_loader)
        else:
            self.eval = EvalLoop(model, train_loader)

    def fit(self, epochs=1):
        for epoch in range(epochs):
            running_loss = 0.0
            total_train = 0
            correct_train = 0

            for i, data in enumerate(self.train_loader):
                inputs, labels = data['x'], data['y']
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, labels)
                loss.backward()
                self.optimizer.step()

                _, predicted = torch.max(outputs.data, 1)
                total_train += inputs.shape[0]
                correct_train += predicted.eq(labels.data).sum().item()
                running_loss += loss.item()

                if i % self.print_every == self.print_every-1:
                    print("Loss: ", running_loss / self.print_every)
                    running_loss = 0.0

            train_accuracy = 100.0 * correct_train/total_train
            val_accuracy = self.eval.predict()
            print("Epoch %d: Training Accuracy = %d%%,  Validation Accuracy = %d%%" % (epoch+1, train_accuracy, val_accuracy))

        print("Finished Training")