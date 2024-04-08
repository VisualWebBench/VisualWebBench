import re

import numpy as np
from rouge import Rouge 

import torch
from torchvision.ops import box_iou


def eval_web_caption(preds, golds, **kwargs):
    assert len(preds) == len(golds)
    for i in range(len(preds)):
        if not preds[i]:
            preds[i] = " "

    rouge = Rouge(metrics=['rouge-1', 'rouge-2', 'rouge-l'])
    scores = rouge.get_scores(preds, golds, avg=True)
    return dict(
        rouge_1=scores['rouge-1']['f'] * 100,
        rouge_2=scores['rouge-2']['f'] * 100,
        rouge_l=scores['rouge-l']['f'] * 100
    )


def eval_heading_ocr(preds, golds, **kwargs):
    assert len(preds) == len(golds)
    for i in range(len(preds)):
        if not preds[i]:
            preds[i] = " "

    rouge = Rouge(metrics=['rouge-1', 'rouge-2', 'rouge-l'])
    scores = rouge.get_scores(preds, golds, avg=True)
    return dict(
        rouge_1=scores['rouge-1']['f'] * 100,
        rouge_2=scores['rouge-2']['f'] * 100,
        rouge_l=scores['rouge-l']['f'] * 100
    )


def eval_element_ocr(preds, golds, **kwargs):
    assert len(preds) == len(golds)
    for i in range(len(preds)):
        if not preds[i] or len(preds[i]) == 1:
            preds[i] = " "

    rouge = Rouge(metrics=['rouge-1', 'rouge-2', 'rouge-l'])
    scores = rouge.get_scores(preds, golds, avg=True)
    return dict(
        rouge_1=scores['rouge-1']['f'] * 100,
        rouge_2=scores['rouge-2']['f'] * 100,
        rouge_l=scores['rouge-l']['f'] * 100
    )


def eval_action_prediction(preds, golds, **kwargs):
    results = []
    for pred, gold in zip(preds, golds):
        cur_pred = parse_multi_choice_response(pred, [chr(ord('A')+i) for i in range(8)])
        try:
            if ord('A') <= ord(cur_pred) <= ord('Z'):
                cur_pred = ord(cur_pred) - ord('A')
            else:
                cur_pred = -1
        except:
            cur_pred = -1
        results.append(cur_pred == gold)

    return dict(
        accuracy=sum(results) / len(results) * 100
    )


def eval_element_ground(preds, golds, **kwargs):
    results = []
    for pred, gold in zip(preds, golds):
        cur_pred = parse_multi_choice_response(pred, [chr(ord('A')+i) for i in range(8)])
        try:
            if ord('A') <= ord(cur_pred) <= ord('Z'):
                cur_pred = ord(cur_pred) - ord('A')
            else:
                cur_pred = -1
        except:
            cur_pred = -1
        results.append(cur_pred == gold)

    return dict(
        accuracy=sum(results) / len(results) * 100
    )


def eval_action_ground(preds, golds, **kwargs):
    results = []
    for pred, gold in zip(preds, golds):
        cur_pred = parse_multi_choice_response(pred, [chr(ord('A')+i) for i in range(8)])
        try:
            if ord('A') <= ord(cur_pred) <= ord('Z'):
                cur_pred = ord(cur_pred) - ord('A')
            else:
                cur_pred = -1
        except:
            cur_pred = -1
        results.append(cur_pred == gold)

    return dict(
        accuracy=sum(results) / len(results) * 100
    )


def eval_element_bbox_ground(preds, golds, **kwargs):
    # print('preds[0]', preds[0])
    # print('golds[0]', golds[0])
    correct = total_cnt = 0
    for i, predict_bbox in enumerate(preds):
        if not predict_bbox:
            predict_bbox = (0., 0., 0., 0.)
        try:
            target_bbox = torch.tensor(golds[i], dtype=torch.float32).view(-1, 4)
            predict_bbox = torch.tensor(predict_bbox, dtype=torch.float32).view(-1, 4)
            iou = box_iou(predict_bbox, target_bbox)
            iou = iou.item()
            if iou >= 0.5:
                correct += 1
        except:
            pass

        total_cnt += 1

    return dict(
        precision=correct / total_cnt * 100
    )


def eval_action_bbox_ground(preds, golds, **kwargs):
    correct = total_cnt = 0
    for i, predict_bbox in enumerate(preds):
        if not predict_bbox:
            predict_bbox = (0., 0., 0., 0.)
        try:
            target_bbox = torch.tensor(golds[i], dtype=torch.float32).view(-1, 4)
            predict_bbox = torch.tensor(predict_bbox, dtype=torch.float32).view(-1, 4)
            iou = box_iou(predict_bbox, target_bbox)
            iou = iou.item()
            if iou >= 0.5:
                correct += 1
        except:
            pass

        total_cnt += 1

    return dict(
        precision=correct / total_cnt * 100
    )


def eval_webqa(preds, golds, **kwargs):
    f1_scores = []
    rouge = Rouge(metrics=['rouge-1'])
    for pred, gold_list in zip(preds, golds):
        try:
            if not pred:
                pred = " "
            cur_f1 = max([rouge.get_scores([pred], [gold], avg=True)['rouge-1']['f'] for gold in gold_list])
            f1_scores.append(cur_f1)
        except:
            pass

    return dict(
        f1=sum(f1_scores) / len(f1_scores) * 100
    )

def eval_element_point_ground(preds, golds):
    acc_lst = []
    for pred, gold in zip(preds, golds):
        x, y = pred
        left, top, right, bottom = gold
        acc_lst.append(left<=x<=right and top<=y<=bottom)
    return dict(
        accuracy=sum(acc_lst) / len(acc_lst) * 100
    )

def eval_action_point_ground(preds, golds):
    acc_lst = []
    for pred, gold in zip(preds, golds):
        x, y = pred
        left, top, right, bottom = gold
        acc_lst.append(left<=x<=right and top<=y<=bottom)
    return dict(
        accuracy=sum(acc_lst) / len(acc_lst) * 100
    )

# ----------- Process Multi-choice -------------
def parse_multi_choice_response(response: str, all_choices):
    """
    Parse the prediction from the generated response.
    Return the predicted index e.g., A, B, C, D.
    """
    if len(response) == 1:
        return response.upper()
    elif not response:
        return 'a'
    elif re.match(r"[A-Z]\.", response):
        return response[0]

    for char in [',', '.', '!', '?', ';', ':', "'", '"']:
        response = response.replace(char, "")
    response = " " + response + " " # add space to avoid partial match

    ans_with_brack = False
    candidates = []
    for choice in all_choices:  # e.g., (A) (B) (C) (D)
        if f'({choice})' in response:
            candidates.append(choice)
            ans_with_brack = True

    if len(candidates) == 0:
        for choice in all_choices: # e.g., A B C D
            if f' {choice} ' in response:
                candidates.append(choice)

    if len(candidates) == 0:  # still not get answer
        # pred_index = random.choice(all_choices)
        pred_index = "z"
    elif len(candidates) > 1:
        start_indexes = []
        if ans_with_brack: 
            for can in candidates:
                index = response.rfind(f'({can})')
                start_indexes.append(index) # -1 will be ignored anyway
            # start_indexes = [generated_response.index(f'({can})') for can in candidates]
        else:
            for can in candidates:
                index = response.rfind(f" {can} ")
                start_indexes.append(index)
        # get the last one
        pred_index = candidates[np.argmax(start_indexes)]
    else: # if only one candidate, use it.
        pred_index = candidates[0]

    return pred_index
