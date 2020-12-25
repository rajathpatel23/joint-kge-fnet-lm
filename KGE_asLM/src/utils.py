from sklearn.utils import shuffle
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, classification_report
from sklearn.metrics import precision_recall_curve, auc
import logging

def get_aucpr(y_out, y_true):
    '''
    :param y_out: predicted value
    :param y_true: True values
    :return: aucpr score
    '''
    precision, recall, threshold = precision_recall_curve(y_out, y_true)
    aucpr = auc(recall, precision)
    return aucpr


def tuning(prediction):
    '''
    The function thresholds the prediction made by the classifier
    :param prediction: list with prediction values between 0 and 1
    :return: list of prediction of score 0 or 1
    '''
    tune_prediction = []
    for i in range(len(prediction)):
        if prediction[i] >= 0.5:
            tune_prediction.append(1)
        else:
            tune_prediction.append(-1)
    return tune_prediction


def tuning_1(prediction):
    '''
    The function thresholds the prediction made by the classifier
    :param prediction: list with prediction values between 0 and 1
    :return: list of prediction of score 0 or 1
    '''
    tune_prediction = []
    for i in range(len(prediction)):
        if prediction[i] >= 0.5:
            tune_prediction.append(1)
        else:
            tune_prediction.append(0)
    return tune_prediction


def evaluation(y_out, pred_out, data_type=False):
    '''
    evaluation function calculate the performance of the classifier
    :param y_out: predicted values
    :param pred_out: true values
    :param data_type: DataSet {"fb13", "WN11", "WN18", "FB15k", FB15"}
    :return: None - print  evaluation results
    '''
    if data_type:
        pred_1 = tuning_1(pred_out[0])
    else:
        pred_1 = tuning(pred_out)
    # for val in pred_out[0]:
    #     logging.info("val => {}".format(val))

    acc = 0
    for i in range(len(pred_1)):
        if pred_1[i] == y_out[i]:
            acc += 1
    b = (acc / len(pred_1)) * 100
    logging.info("The test accuracy: {}".format(b))

    roc_score_test = roc_auc_score(y_out, pred_out)
    aucpr_test = get_aucpr(y_out, pred_out)
    f1_test = f1_score(y_out, pred_1)
    precision_test = precision_score(y_out, pred_1)
    recall_test = recall_score(y_out, pred_1)
    logging.info("This is the AUCPR score: {}".format(aucpr_test))
    logging.info("This is the roc_score: {}".format(roc_score_test))
    logging.info("This is the F1 score: {}".format(f1_test))
    logging.info("This is precision: {}".format(precision_test))
    logging.info("This is recall: {}".format(recall_test))
    logging.info("Classification report: {}".format(classification_report(y_out, pred_1)))
    

def get_ids(head, word_2_id):
    """
    :param head:  entities
    :param word_2_id: word to 2 dict
    :return: list of ids to corresponding head entity
    """
    temp_list = []
    en = head.split("_")
    for k in en:
        if k in word_2_id.keys():
            temp_list.append(word_2_id[k])

        else:
            temp_list.append(word_2_id['unk'])
    return temp_list


def eval_data(model, head_vec, tail_vec, rel_vec, label_ids, data_type=False):
    N = len(head_vec)
    g1 = 1024
    output_dev_collector = []
    for k1 in (range(0, N, g1)):
        if k1+ g1 > N:
            d1 = N
        else:
            d1 = k1 + g1

        head, tail, rel, embed = model.get_embed(head_vec[k1:d1], tail_vec[k1:d1], rel_vec[k1:d1])
        output_= model.predict(embed)
        output_dev_collector += output_[0].tolist()
    evaluation(label_ids, output_dev_collector, data_type)
