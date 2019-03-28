# coding: utf-8
# created by deng on 2019-02-13

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report

from dataset import ExhaustiveDataset
from utils.torch_util import calc_f1
from utils.path_util import from_project_root


def evaluate(model, data_url):
    """ eval model on specific dataset

    Args:
        model: model to evaluate
        data_url: url to data for evaluating

    Returns:
        metrics on dataset

    """
    print("\nEvaluating model use data from ", data_url, "\n")
    max_region = model.max_region
    dataset = ExhaustiveDataset(data_url, next(model.parameters()).device, max_region=max_region)
    data_loader = DataLoader(dataset, batch_size=100, collate_fn=dataset.collate_func)
    model.eval()
    region_true_list = list()
    region_pred_list = list()
    # switch to eval mode
    with torch.no_grad():
        for data, labels, records_list in data_loader:
            batch_region_labels = torch.argmax(model.forward(*data), dim=1).cpu()
            lengths = data[1]
            for region_labels, length, true_records in zip(batch_region_labels, lengths, records_list):
                pred_records = {}
                ind = 0
                for region_size in range(1, max_region + 1):
                    for start in range(0, lengths[0] - region_size + 1):
                        if 0 < region_labels[ind] < 6:
                            pred_records[(start, start + region_size)] = region_labels[ind]
                        ind += 1

                for region in true_records:
                    true_label = dataset.label_ids[true_records[region]]
                    pred_label = pred_records[region] if region in pred_records else 0
                    region_true_list.append(true_label)
                    region_pred_list.append(pred_label)
                for region in pred_records:
                    if region not in true_records:
                        region_pred_list.append(pred_records[region])
                        region_true_list.append(0)

    print(classification_report(region_true_list, region_pred_list,
                                target_names=list(dataset.label_ids)[:6], digits=6))

    ret = dict()
    tp = fp = fn = 0
    for pv, tv in zip(region_pred_list, region_true_list):
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
