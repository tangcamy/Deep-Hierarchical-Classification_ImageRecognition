'''
Graph plotting functions.
'''

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

fig = plt.figure(figsize=(20, 5))

def plot_loss_acc(path, num_epoch, train_accuracies_superclass, train_accuracies_subclass, train_losses,
                    test_accuracies_superclass, test_accuracies_subclass, test_losses):
    '''
    Plot line graphs for the accuracies and loss at every epochs for both training and testing.
    '''

    plt.clf()

    epochs = [x for x in range(num_epoch[0]+1)]

    train_superclass_accuracy_df = pd.DataFrame({"Epochs":epochs, "Accuracy":train_accuracies_superclass[0], "Mode":['train']*(num_epoch[0]+1)})
    train_subclass_accuracy_df = pd.DataFrame({"Epochs":epochs, "Accuracy":train_accuracies_subclass[0], "Mode":['train']*(num_epoch[0]+1)})
    test_superclass_accuracy_df = pd.DataFrame({"Epochs":epochs, "Accuracy":test_accuracies_superclass[0], "Mode":['test']*(num_epoch[0]+1)})
    test_subclass_accuracy_df = pd.DataFrame({"Epochs":epochs, "Accuracy":test_accuracies_subclass[0], "Mode":['test']*(num_epoch[0]+1)})

    data_superclass = pd.concat([train_superclass_accuracy_df, test_superclass_accuracy_df])
    data_subclass = pd.concat([train_subclass_accuracy_df, test_subclass_accuracy_df])
  
    sns.lineplot(data=data_superclass.reset_index(drop=True), x='Epochs', y='Accuracy', hue='Mode')
    plt.title('Superclass Accuracy Graph')
    plt.savefig(path[0]+f'accuracy_superclass_epoch.png')
    plt.show()
    plt.clf()

    sns.lineplot(data=data_subclass.reset_index(drop=True), x='Epochs', y='Accuracy', hue='Mode')
    plt.title('Subclass Accuracy Graph')
    plt.savefig(path[0]+f'accuracy_subclass_epoch.png')
    plt.show()
    plt.clf()


    #--計算LOSS--#
    train_loss_df = pd.DataFrame({"Epochs":epochs, "Loss":train_losses[0], "Mode":['train']*(num_epoch[0]+1)})
    test_loss_df = pd.DataFrame({"Epochs":epochs, "Loss":test_losses[0], "Mode":['test']*(num_epoch[0]+1)})

    train_test_loss = pd.concat([train_loss_df, test_loss_df])

    sns.lineplot(data=train_test_loss.reset_index(drop=True), x='Epochs', y='Loss', hue='Mode')
    plt.title('Loss Graph')
    plt.savefig(path[0]+f'loss_epoch.png')
    plt.show()
    plt.clf()
    return None
