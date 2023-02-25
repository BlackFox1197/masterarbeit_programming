import torch
from matplotlib import pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader


def evaluate(device, model, dataset, batch_size):
    loader = DataLoader(dataset, shuffle=False, batch_size=batch_size)
    true, preds = [], []
    with torch.no_grad():
        for batch, (X, z) in enumerate(loader):
            X, z = X.to(device),  z.to(device)
            pred = model(X)
            labels = [torch.squeeze(a.nonzero()).item() for a in z]
            true = true + labels

            preds = preds + pred.argmax(1).cpu().numpy().tolist()
            #preds.append(pred.argmax(1))
            if batch % 10 == 0:
                print(f"{batch*batch_size} of {len(dataset)}")

    return true, preds




def show_bars(predictions, label_list, color="r", title_string = ""):

    fig = plt.figure(constrained_layout=True , figsize=(10, 20))
    fig.suptitle(title_string, fontsize=21)

    # create 3x1 subfigs
    subfigs = fig.subfigures(nrows=7, ncols=1)
    for label in range(len(label_list)):

        axs = subfigs[label].subplots(nrows=1, ncols=4)

        subfigs[label].suptitle(f'{label_list[label]}', fontsize=19)
        sns.histplot([item[0] for item in predictions[label_list[label]]], bins=25, color=color, ax=axs[0], kde=True, kde_kws={"cut": 3}, stat="density")
        axs[0].yaxis.label.set_visible(False)
        axs[0].set_title(f'Parameter 0')
        sns.histplot([item[1] for item in predictions[label_list[label]]], bins=25, color=color, ax=axs[1], kde=True, kde_kws={"cut": 3}, stat="density")
        axs[1].yaxis.label.set_visible(False)
        axs[1].set_title(f'Parameter 1')
        sns.histplot([item[2] for item in predictions[label_list[label]]], bins=25, color=color, ax=axs[2], kde=True, kde_kws={"cut": 3}, stat="density")
        axs[2].yaxis.label.set_visible(False)
        axs[2].set_title(f'Parameter 2')
        sns.histplot([item[3] for item in predictions[label_list[label]]], bins=25, color=color, ax=axs[3], kde=True, kde_kws={"cut": 3}, stat="density")
        axs[3].yaxis.label.set_visible(False)
        axs[3].set_title(f'Parameter 3')
