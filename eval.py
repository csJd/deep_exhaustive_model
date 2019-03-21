# coding: utf-8
# created by deng on 2019-02-13

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report

from dataset import ExhaustiveDataset
from utils.torch_util import calc_f1
from utils.path_util import from_project_root


def evaluate(model, data_url, max_region=10, show_padding=False):
    """ eval model on specific dataset

    Args:
        model: model to evaluate
        data_url: url to data for evaluating
        max_region: max_region for ExhaustiveDataset
        show_padding: show padding regions count in classification report

    Returns:
        metrics on dataset

    """
    print("\nEvaluating model use data from ", data_url, "\n")
    dataset = ExhaustiveDataset(data_url, next(model.parameters()).device, max_region=max_region)
    data_loader = DataLoader(dataset, batch_size=100, collate_fn=dataset.collate_func)
    model.eval()
    # switch to eval mode
    with torch.no_grad():
        true_list = list()
        pred_list = list()
        for data, labels in data_loader:
            pred = torch.argmax(model.forward(*data), dim=1)
            pred = pred.view(-1).cpu()
            true = labels.view(-1).cpu()
            for pv, tv in zip(pred, true):
                if tv == 6 and not show_padding:
                    continue
                true_list.append(tv)
                pred_list.append(pv)

        # print report on dataset
        target_names = list(dataset.label_ids)
        if not show_padding:
            target_names = target_names[:6]
        print(classification_report(true_list, pred_list, target_names=target_names, digits=6))

        ret = dict()
        tp = fp = fn = 0
        for pv, tv in zip(pred_list, true_list):
            # if for padding
            if tv == 6 or pv == tv == 0:
                continue
            if pv == tv:
                tp += 1
            else:  # predict value != true value
                if pv > 0:
                    fp += 1
                if tv > 0:
                    fn += 1

        ret['precision'], ret['recall'], ret['f1'] = calc_f1(tp, fp, fn)
        return ret


def main():
    model_url = from_project_root("data/model/model.pt")
    test_url = from_project_root("data/genia.test.iob2")
    model = torch.load(model_url)
    evaluate(model, test_url)
    pass


if __name__ == '__main__':
    main()
