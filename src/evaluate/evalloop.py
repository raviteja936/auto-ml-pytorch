import torch

class EvalLoop:
    def __init__(self, model, dataloader):
        self.model = model
        self.dataloader = dataloader

    def predict(self):
        total = 0
        correct = 0
        nb_classes = 2
        confusion_matrix = torch.zeros(nb_classes, nb_classes)
        with torch.no_grad():
            for data in self.dataloader:
                inputs, labels = data['x'], data['y']
                outputs = self.model(inputs)
                # print(outputs.data)
                _, predicted = torch.max(outputs.data, 1)
                total += inputs.shape[0]
                correct += predicted.eq(labels.data).sum().item()

                for t, p in zip(labels.view(-1), predicted.view(-1)):
                    confusion_matrix[t, p] += 1

            accuracy = 100.0 * correct / total
            # self.writer.add_scalar('val accuracy', accuracy, epoch)
            print(confusion_matrix)
        return accuracy
