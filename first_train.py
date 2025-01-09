# in the name of God
#
import numpy as np
import torch

def compute_iou(box1, box2):
    
    x_left = max(box1[0], box2[0])
    y_top    = max(box1[1], box2[1])
    x_right  = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    iou = intersection_area / float(box1_area + box2_area - intersection_area)
    
    return iou


def average_precision(recall, precision):
    
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
    
    indices = np.where(mrec[1:] != mrec[:-1])[0]
    
    ap = np.sum((mrec[indices + 1] - mrec[indices]) * mpre[indices + 1])
    
    return ap


def evaluate_model_custom(model, data_loader, device, iou_threshold=0.5, score_threshold=0.05, max_detections=100):
    """
    model torch.nn.Module
    iou_threshold : float, optional threshold to consider a detection as True Positive. 0.5.
    score_threshold : float, optional Minimum score for detections to consider. 0.05
    max_detections : int, 100.
    """
    
    model.eval()
    
    all_predictions = []
    all_ground_truths = {}
    
    with torch.no_grad():
        
        for batch_idx, (images, targets) in enumerate(data_loader):
            
            images = list(img.to(device) for img in images)
            outputs = model(images)
            
            for i, output in enumerate(outputs):
                
                image_id = targets[i]['image_id'].item()
                
                # GT
                gt_boxes = targets[i]['boxes'].cpu().numpy()
                gt_labels = targets[i]['labels'].cpu().numpy()
                
                if image_id not in all_ground_truths:
                    all_ground_truths[image_id] = []
                
                for gt_box, gt_label in zip(gt_boxes, gt_labels):
                    all_ground_truths[image_id].append({
                        'bbox': gt_box,
                        'label': gt_label,
                        'matched': False
                    })
                
                # pred
                pred_boxes = output.get('boxes', None)
                pred_scores = output.get('scores', None)
                pred_labels = output.get('labels', None)
                
                if pred_boxes is None or pred_scores is None or pred_labels is None:
                    continue
                
                keep = pred_scores >= score_threshold
                
                pred_boxes = pred_boxes[keep]
                pred_scores = pred_scores[keep]
                pred_labels = pred_labels[keep]
                
                if len(pred_scores) > max_detections:
                    pred_boxes = pred_boxes[:max_detections]
                    pred_scores = pred_scores[:max_detections]
                    pred_labels = pred_labels[:max_detections]
                
                for box, score, label in zip(pred_boxes, pred_scores, pred_labels):
                    
                    box = box.cpu().numpy()
                    score = score.cpu().item()
                    label = label.cpu().item()
                    
                    all_predictions.append({
                        'image_id': image_id,
                        'bbox': box,
                        'score': score,
                        'label': label
                    })
    
    predictions_by_class = {}
    for pred in all_predictions:
        
        cls = pred['label']
        
        if cls not in predictions_by_class:
            predictions_by_class[cls] = []
        
        predictions_by_class[cls].append(pred)
    
    for cls in predictions_by_class:
        
        predictions_by_class[cls] = sorted(predictions_by_class[cls], key=lambda x: x['score'], reverse=True)
    
    all_classes = list(predictions_by_class.keys())
    
    for gt_list in all_ground_truths.values():
        
        for gt in gt_list:
            
            if gt['label'] not in all_classes:
                all_classes.append(gt['label'])
    
    average_precisions = []
    
    for cls in all_classes:
        
        preds = predictions_by_class.get(cls, [])
        num_preds = len(preds)
        
        gt_for_class = {}
        num_gt = 0
        
        for image_id, gt_list in all_ground_truths.items():
            
            gt_boxes = [gt for gt in gt_list if gt['label'] == cls]
            if len(gt_boxes) > 0:
                gt_for_class[image_id] = gt_boxes
                num_gt += len(gt_boxes)
        
        if num_gt == 0:
            average_precisions.append(0)
            continue
        
        TP = np.zeros(num_preds)
        FP = np.zeros(num_preds)
        
        for idx, pred in enumerate(preds):
            
            image_id = pred['image_id']
            pred_box = pred['bbox']
            max_iou = 0
            max_gt_idx = -1
            
            if image_id in gt_for_class:
                gt_boxes = gt_for_class[image_id]
                for gt_idx, gt in enumerate(gt_boxes):
                    iou = compute_iou(pred_box, gt['bbox'])
                    if iou > max_iou:
                        max_iou = iou
                        max_gt_idx = gt_idx
            if max_iou >= iou_threshold:
                if not gt_for_class[image_id][max_gt_idx]['matched']:
                    TP[idx] = 1
                    gt_for_class[image_id][max_gt_idx]['matched'] = True
                else:
                    FP[idx] = 1
            else:
                FP[idx] = 1
        
        cumulative_TP = np.cumsum(TP)
        cumulative_FP = np.cumsum(FP)
        
        precision = cumulative_TP / (cumulative_TP + cumulative_FP + 1e-6)
        recall = cumulative_TP / (num_gt + 1e-6)
        
        ap = average_precision(recall, precision)
        average_precisions.append(ap)
    
    mAP = np.mean(average_precisions)
    
    return mAP

#cloner174