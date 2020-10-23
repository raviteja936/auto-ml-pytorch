import sys
# import tensorflow as tf
# from matplotlib.plt import pyplot
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from utils.cli import CliArgs
from pipes.pipeline import DataPipe


def experiment(args):
    args = CliArgs(args)
    params = args.get_params()
    pipe = DataPipe(params, "train")
    train_data, val_data = pipe.build()
    print(len(train_data), len(val_data))
    # model = Model(params)
    # model.compile(optimizer='adam', loss=tf.keras.losses.MeanSquaredError(), metrics=['mse'])
    # history = model.fit(train_data, epochs=20)
    # predictions = model.predict(test_data)

    # Show some results
    # for prediction, actual in zip(predictions[:10], list(test_data)[0][1][:10]):
    #     print("Predicted outcome: ", prediction[0], " | Actual outcome: ", actual.numpy())
    # plot metrics
    # pyplot.plot(history.history['mse'])
    # pyplot.show()

    return


if __name__ == "__main__":
    experiment(sys.argv[1:])
