# coding: utf-8
# created by deng on 2019-02-13

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report

from dataset import ExhaustiveDataset, gen_sentence_tensors
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
    # switch to eval mode
    model.eval()

    region_true_list = list()
    region_pred_list = list()
    region_true_count = 0
    region_pred_count = 0

    with torch.no_grad():
        for data, labels, records_list in data_loader:
            batch_region_labels = torch.argmax(model.forward(*data), dim=1).cpu()
            lengths = data[1]
            batch_maxlen = lengths[0]
            for region_labels, length, true_records in zip(batch_region_labels, lengths, records_list):
                pred_records = {}
                ind = 0
                for region_size in range(1, max_region + 1):
                    for start in range(0, batch_maxlen - region_size + 1):
                        end = start + region_size
                        if 0 < region_labels[ind] < dataset.n_tags and end <= length:
                            pred_records[(start, start + region_size)] = region_labels[ind]
                        ind += 1

                region_true_count += len(true_records)
                region_pred_count += len(pred_records)

                for region in true_records:
                    true_label = dataset.label_list.index(true_records[region])
                    pred_label = pred_records[region] if region in pred_records else 0
                    region_true_list.append(true_label)
                    region_pred_list.append(pred_label)
                for region in pred_records:
                    if region not in true_records:
                        region_pred_list.append(pred_records[region])
                        region_true_list.append(0)

    print(classification_report(region_true_list, region_pred_list,
                                target_names=dataset.label_list, digits=6))

    ret = dict()
    tp = 0
    for pv, tv in zip(region_pred_list, region_true_list):
        if pv == tv:
            tp += 1
    fp = region_pred_count - tp
    fn = region_true_count - tp

    ret['precision'], ret['recall'], ret['f1'] = calc_f1(tp, fp, fn)
    return ret


def predict(model, sentences, labels):
    """ predict NER result for sentence list

    Args:
        model: trained model
        sentences: sentences to be predicted

    Returns:
        predicted results

    """
    max_region = model.max_region
    device = next(model.parameters()).device
    tensors = gen_sentence_tensors(
        sentences, device, from_project_root('data/vocab.json'))
    pred_regions_list = torch.argmax(model.forward(*tensors), dim=1).cpu()

    lengths = tensors[1]
    pred_sentence_records = []
    for pred_regions, length in zip(pred_regions_list, lengths):
        pred_records = {}
        ind = 0
        for region_size in range(1, max_region + 1):
            for start in range(0, lengths[0] - region_size + 1):
                if 0 < pred_regions[ind] < len(labels):
                    pred_records[(start, start + region_size)] = \
                        labels[pred_regions[ind]]
                ind += 1
        pred_sentence_records.append(pred_records)
    return pred_sentence_records


def predict_on_iob2(model, iob_url):
    """ predict on iob2 file and save the results

    Args:
        model: trained model
        iob_url: url to iob file

    """

    save_url = iob_url.replace('.iob2', '.pred.txt')
    print("predicting on {} \n the result will be saved in {}".format(
        iob_url, save_url))
    test_set = ExhaustiveDataset(iob_url, device=next(
        model.parameters()).device)

    model.eval()
    with open(save_url, 'w', encoding='utf-8', newline='\n') as save_file:
        for sentence, records in test_set:
            save_file.write(' '.join(sentence) + '\n')
            save_file.write("length = {} \n".format(len(sentence)))
            save_file.write("Gold: {}\n".format(str(records)))
            pred_result = str(predict(model, [sentence], test_set.label_list)[0])
            save_file.write("Pred: {}\n\n".format(pred_result))


def main():
    model_url = from_project_root("data/model/model.pt")
    print("loading model from", model_url)
    # model = torch.load(model_url, map_location='cpu')
    model = torch.load(model_url)
    test_url = from_project_root("data/genia.test.iob2")
    evaluate(model, test_url)
    # predict_on_iob2(model, test_url)
    pass


if __name__ == '__main__':
    main()
