import torch
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from torchvision.ops import box_iou


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


def match_predictions_to_ground_truths(predictions, ground_truths, iou_threshold=0.5):
    
    matched_preds = []
    
    unmatched_gts = ground_truths.copy()
    pred_boxes = predictions['boxes'].cpu().numpy()
    pred_labels = predictions['labels'].cpu().numpy()
    pred_scores = predictions['scores'].cpu().numpy()
    gt_boxes = [gt['bbox'] for gt in ground_truths]
    gt_labels = [gt['label'] for gt in ground_truths]
    gt_matched = [False] * len(ground_truths)
    
    sorted_indices = np.argsort(-pred_scores)
    pred_boxes = pred_boxes[sorted_indices]
    pred_labels = pred_labels[sorted_indices]
    pred_scores = pred_scores[sorted_indices]
    
    for pred_box, pred_label, pred_score in zip(pred_boxes, pred_labels, pred_scores):
        
        best_iou = 0
        best_gt_idx = -1
        for gt_idx, (gt_box, gt_label) in enumerate(zip(gt_boxes, gt_labels)):
            
            if gt_matched[gt_idx]:
                continue
            
            if pred_label != gt_label:
                continue
            
            iou = compute_iou(pred_box, gt_box)
            
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx
        
        if best_iou >= iou_threshold:
            
            matched_preds.append({
                'pred_box': pred_box,
                'pred_label': pred_label,
                'pred_score': pred_score,
                'gt_box': gt_boxes[best_gt_idx],
                'gt_label': gt_labels[best_gt_idx],
                'iou': best_iou
            })
            gt_matched[best_gt_idx] = True
        
        else:
            matched_preds.append({
                'pred_box': pred_box,
                'pred_label': pred_label,
                'pred_score': pred_score,
                'gt_box': None,
                'gt_label': None,
                'iou': 0.0
            })
    
    for gt_idx, matched in enumerate(gt_matched):
        if not matched:
            unmatched_gts.append({
                'bbox': gt_boxes[gt_idx],
                'label': gt_labels[gt_idx]
            })
    
    return matched_preds, unmatched_gts


def evaluate_model_metrics(model, data_loader, device, iou_threshold=0.5, score_threshold=0.05, max_detections=100):
    
    model.eval()
    
    TP = 0
    FP = 0
    FN = 0
    with torch.no_grad():
        
        for batch_idx, (images, targets) in enumerate(tqdm(data_loader, desc="Evaluating Metrics")):
            
            images = list(img.to(device) for img in images)
            outputs = model(images)
            
            for i, output in enumerate(outputs):
                
                image_id = targets[i]['image_id'].item()
                gt_boxes = targets[i]['boxes'].cpu().numpy()
                gt_labels = targets[i]['labels'].cpu().numpy()
                ground_truths = []
                
                for gt_box, gt_label in zip(gt_boxes, gt_labels):
                    ground_truths.append({
                        'bbox': gt_box,
                        'label': gt_label
                    })
                
                predictions = {
                    'boxes': output['boxes'],
                    'labels': output['labels'],
                    'scores': output['scores']
                }
                
                high_score_idxs = predictions['scores'] >= score_threshold
                predictions['boxes'] = predictions['boxes'][high_score_idxs][:max_detections]
                predictions['labels'] = predictions['labels'][high_score_idxs][:max_detections]
                predictions['scores'] = predictions['scores'][high_score_idxs][:max_detections]
                
                matched_preds, unmatched_gts = match_predictions_to_ground_truths(predictions, ground_truths, iou_threshold)
                
                for pred in matched_preds:
                    if pred['gt_box'] is not None:
                        TP += 1
                    else:
                        FP += 1
                
                FN += len(unmatched_gts)
    
    precision = TP / (TP + FP + 1e-6)
    recall = TP / (TP + FN + 1e-6)
    f1_score = 2 * (precision * recall) / (precision + recall + 1e-6)
    
    metrics = {
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1_score
    }
    
    return metrics


def calculate_ap(recalls, precisions):
    
    sorted_indices = np.argsort(recalls)
    recalls = np.array(recalls)[sorted_indices]
    precisions = np.array(precisions)[sorted_indices]
    
    recalls = np.concatenate(([0.0], recalls, [1.0]))
    precisions = np.concatenate(([0.0], precisions, [0.0]))
    
    for i in range(len(precisions) - 1, 0, -1):
        precisions[i - 1] = max(precisions[i - 1], precisions[i])
    
    indices = np.where(recalls[1:] != recalls[:-1])[0]
    ap = np.sum((recalls[indices + 1] - recalls[indices]) * precisions[indices + 1])
    
    return ap


def evaluate_map(model, data_loader, device, iou_threshold=0.5, score_threshold=0.05, max_detections=100, num_classes=10):
    
    model.eval()
    
    ap_dict = {cls: 0.0 for cls in range(1, num_classes + 1)}
    total_gt_per_class = {cls: 0 for cls in range(1, num_classes + 1)}
    tp_fp_data = {cls: {'scores': [], 'tp': [], 'fp': []} for cls in range(1, num_classes + 1)}
    
    with torch.no_grad():
        
        for batch_idx, (images, targets) in enumerate(tqdm(data_loader, desc="Evaluating mAP")):
            
            images = list(img.to(device) for img in images)
            outputs = model(images)
            
            for i, output in enumerate(outputs):
                
                image_id = targets[i]['image_id'].item()
                gt_boxes = targets[i]['boxes'].cpu().numpy()
                gt_labels = targets[i]['labels'].cpu().numpy()
                pred_boxes = output['boxes'].cpu().numpy()
                pred_scores = output['scores'].cpu().numpy()
                pred_labels = output['labels'].cpu().numpy()
                
                high_score_idxs = pred_scores >= score_threshold
                pred_boxes = pred_boxes[high_score_idxs][:max_detections]
                pred_scores = pred_scores[high_score_idxs][:max_detections]
                pred_labels = pred_labels[high_score_idxs][:max_detections]
                
                for cls in range(1, num_classes + 1):
                    
                    gt_boxes_cls = gt_boxes[gt_labels == cls]
                    total_gt_per_class[cls] += len(gt_boxes_cls)
                    if len(gt_boxes_cls) == 0:
                        continue
                    
                    preds_cls = pred_boxes[pred_labels == cls]
                    scores_cls = pred_scores[pred_labels == cls]
                    matched_gt = [False] * len(gt_boxes_cls)
                    
                    for pred_box, score in zip(preds_cls, scores_cls):
                        
                        iou_max = 0
                        best_match_idx = -1
                        
                        for gt_idx, gt_box in enumerate(gt_boxes_cls):
                            
                            if matched_gt[gt_idx]:
                                continue
                            
                            iou = compute_iou(pred_box, gt_box)
                            
                            if iou > iou_max:
                                iou_max = iou
                                best_match_idx = gt_idx
                        
                        if iou_max >= iou_threshold:
                            matched_gt[best_match_idx] = True
                            tp_fp_data[cls]['tp'].append(1)
                            tp_fp_data[cls]['fp'].append(0)
                        
                        else:
                            tp_fp_data[cls]['tp'].append(0)
                            tp_fp_data[cls]['fp'].append(1)
                        
                        tp_fp_data[cls]['scores'].append(score)
    
    for cls in range(1, num_classes + 1):
        
        if total_gt_per_class[cls] == 0:
            ap_dict[cls] = 0.0
            continue
        
        scores = np.array(tp_fp_data[cls]['scores'])
        tp = np.array(tp_fp_data[cls]['tp'])
        fp = np.array(tp_fp_data[cls]['fp'])
        sorted_indices = np.argsort(-scores)
        tp = np.cumsum(tp[sorted_indices])
        fp = np.cumsum(fp[sorted_indices])
        precision = tp / (tp + fp + 1e-6)
        recall = tp / (total_gt_per_class[cls] + 1e-6)
        ap = calculate_ap(recall, precision)
        ap_dict[cls] = ap
    
    mAP = np.mean(list(ap_dict.values()))
    
    return ap_dict, mAP


def all_evaluation(model, data_loader, device, dataset, iou_threshold=0.5, score_threshold=0.05, max_detections=100):
    
    metrics = evaluate_model_metrics(
        model=model,
        data_loader=data_loader,
        device=device,
        iou_threshold=iou_threshold,
        score_threshold=score_threshold,
        max_detections=max_detections
    )
    
    ap_dict, mAP = evaluate_map(
        model=model,
        data_loader=data_loader,
        device=device,
        iou_threshold=iou_threshold,
        score_threshold=score_threshold,
        max_detections=max_detections,
        num_classes=len(dataset.class_to_idx) - 1
    )
    
    results = {
        'Precision': metrics['Precision'],
        'Recall': metrics['Recall'],
        'F1 Score': metrics['F1 Score'],
        'AP_per_class': ap_dict,
        'mAP': mAP
    }
    
    return results


def plot_pr_curve(model, data_loader, device, dataset, iou_threshold=0.5, score_threshold=0.05, max_detections=100):
    
    model.eval()
    
    all_scores = []
    all_labels = []
    all_preds = []
    all_gt = []
    with torch.no_grad():
        
        for batch_idx, (images, targets) in enumerate(tqdm(data_loader, desc="Collecting PR Curve Data")):
            
            images = list(img.to(device) for img in images)
            outputs = model(images)
            for i, output in enumerate(outputs):
                
                image_id = targets[i]['image_id'].item()
                gt_boxes = targets[i]['boxes'].cpu().numpy()
                gt_labels = targets[i]['labels'].cpu().numpy()
                
                for gt_box, gt_label in zip(gt_boxes, gt_labels):
                    all_gt.append({'image_id': image_id, 'bbox': gt_box, 'label': gt_label, 'matched': False})
                
                pred_scores = output['scores'].cpu().numpy()
                pred_labels = output['labels'].cpu().numpy()
                pred_boxes = output['boxes'].cpu().numpy()
                
                high_score_idxs = pred_scores >= score_threshold
                pred_boxes = pred_boxes[high_score_idxs][:max_detections]
                pred_labels = pred_labels[high_score_idxs][:max_detections]
                pred_scores = pred_scores[high_score_idxs][:max_detections]
                
                sorted_indices = np.argsort(-pred_scores)
                pred_boxes = pred_boxes[sorted_indices]
                pred_labels = pred_labels[sorted_indices]
                pred_scores = pred_scores[sorted_indices]
                
                for pred_box, pred_label, pred_score in zip(pred_boxes, pred_labels, pred_scores):
                    
                    all_scores.append(pred_score)
                    all_labels.append(1)
                    all_preds.append({
                        'image_id': image_id,
                        'bbox': pred_box,
                        'label': pred_label,
                        'score': pred_score,
                        'matched': False
                    })
    
    sorted_indices = np.argsort(-np.array(all_scores))
    all_preds = [all_preds[i] for i in sorted_indices]
    all_labels = np.array(all_labels)[sorted_indices]
    all_scores = np.array(all_scores)[sorted_indices]
    TP = 0
    FP = 0
    precisions = []
    recalls = []
    for pred in all_preds:
        
        image_id = pred['image_id']
        pred_box = pred['bbox']
        pred_label = pred['label']
        score = pred['score']
        matched = False
        best_iou = 0.0
        best_gt_idx = -1
        for idx, gt in enumerate(all_gt):
            
            if gt['image_id'] != image_id:
                continue
            if gt['label'] != pred_label:
                continue
            if gt['matched']:
                continue
            iou = compute_iou(pred_box, gt['bbox'])
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = idx

        if best_iou >= iou_threshold:
            TP += 1
            all_gt[best_gt_idx]['matched'] = True
        else:
            FP += 1

        precision = TP / (TP + FP + 1e-6)
        recall = TP / (len([gt for gt in all_gt if gt['label'] == pred_label]) + 1e-6)

        precisions.append(precision)
        recalls.append(recall)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recalls, precisions, marker='.')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Overall Precision-Recall Curve')
    plt.grid(True)
    plt.show()


def plot_pr_curves(pr_curves, dataset, num_classes=10):
    
    plt.figure(figsize=(12, 8))
    for cls in range(1, num_classes):
        
        precision = pr_curves[cls]['precision']
        recall = pr_curves[cls]['recall']
        
        if len(precision) == 0 and len(recall) == 0:
            continue
        class_name = dataset.idx_to_class.get(cls, f"Class_{cls}")
        plt.plot(recall, precision, label=class_name)
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves')
    plt.legend()
    plt.grid(True)
    plt.show()


def evaluate_pr_curve(model, data_loader, device, iou_threshold=0.5, max_detections=100, num_classes=10):
    
    model.eval()
    
    pr_data = {cls: {'scores': [], 'tp': [], 'fp': []} for cls in range(1, num_classes + 1)}
    num_gt = {cls: 0 for cls in range(1, num_classes + 1)}
    gt_records = {cls: [] for cls in range(1, num_classes + 1)}
    
    with torch.no_grad():
        
        for batch_idx, (images, targets) in enumerate(tqdm(data_loader, desc="Collecting PR Curve Data")):
            
            images = list(img.to(device) for img in images)
            outputs = model(images)
            
            for i, output in enumerate(outputs):
                
                image_id = targets[i]['image_id'].item()
                gt_boxes = targets[i]['boxes'].cpu().numpy()
                gt_labels = targets[i]['labels'].cpu().numpy()
                
                for gt_box, gt_label in zip(gt_boxes, gt_labels):
                    gt_records[gt_label].append({
                        'image_id': image_id,
                        'bbox': gt_box,
                        'matched': False
                    })
                    num_gt[gt_label] += 1
                
                predictions = {
                    'boxes': output['boxes'],
                    'labels': output['labels'],
                    'scores': output['scores']
                }
                
                sorted_indices = np.argsort(-predictions['scores'].cpu().numpy())
                predictions['boxes'] = predictions['boxes'][sorted_indices]
                predictions['labels'] = predictions['labels'][sorted_indices]
                predictions['scores'] = predictions['scores'][sorted_indices]
                
                predictions['boxes'] = predictions['boxes'][:max_detections]
                predictions['labels'] = predictions['labels'][:max_detections]
                predictions['scores'] = predictions['scores'][:max_detections]
                for pred_box, pred_label, pred_score in zip(predictions['boxes'], predictions['labels'], predictions['scores']):
                    
                    cls = pred_label.item()
                    
                    pr_data[cls]['scores'].append(pred_score.item())
                    matched = False
                    best_iou = 0
                    best_gt_idx = -1
                    for idx, gt in enumerate(gt_records[cls]):
                        if gt['image_id'] != image_id:
                            continue
                        if gt['matched']:
                            continue
                        iou = compute_iou(pred_box.cpu().numpy(), gt['bbox'])
                        if iou > best_iou:
                            best_iou = iou
                            best_gt_idx = idx
                    
                    if best_iou >= iou_threshold:
                        matched = True
                        gt_records[cls][best_gt_idx]['matched'] = True
                    
                    if matched:
                        pr_data[cls]['tp'].append(1)
                        pr_data[cls]['fp'].append(0)
                    
                    else:
                        pr_data[cls]['tp'].append(0)
                        pr_data[cls]['fp'].append(1)
    
    pr_curves = {}
    for cls in range(1, num_classes + 1):
        
        if len(pr_data[cls]['scores']) == 0:
            pr_curves[cls] = {'precision': [], 'recall': []}
            continue
        
        sorted_indices = np.argsort(-np.array(pr_data[cls]['scores']))
        tp_sorted = np.array(pr_data[cls]['tp'])[sorted_indices]
        fp_sorted = np.array(pr_data[cls]['fp'])[sorted_indices]
        
        cum_tp = np.cumsum(tp_sorted)
        cum_fp = np.cumsum(fp_sorted)
        
        precision = cum_tp / (cum_tp + cum_fp + 1e-6)
        recall = cum_tp / (num_gt[cls] + 1e-6)
        pr_curves[cls] = {
            'precision': precision,
            'recall': recall
        }
    
    return pr_curves


def evaluate_per_class_metrics(model, data_loader, device, iou_threshold=0.5, score_threshold=0.05, max_detections=100, num_classes=10):
    
    model.eval()
    
    tp = {cls: 0 for cls in range(1, num_classes + 1)}
    fp = {cls: 0 for cls in range(1, num_classes + 1)}
    fn = {cls: 0 for cls in range(1, num_classes + 1)}
    
    gt_records = {cls: [] for cls in range(1, num_classes + 1)}
    
    with torch.no_grad():
        
        for batch_idx, (images, targets) in enumerate(tqdm(data_loader, desc="Evaluating Per-Class Metrics")):
            images = list(img.to(device) for img in images)
            outputs = model(images)
            
            for i, output in enumerate(outputs):
                image_id = targets[i]['image_id'].item()
                gt_boxes = targets[i]['boxes'].cpu().numpy()
                gt_labels = targets[i]['labels'].cpu().numpy()
                
                for gt_box, gt_label in zip(gt_boxes, gt_labels):
                    gt_records[gt_label].append({
                        'image_id': image_id,
                        'bbox': gt_box,
                        'matched': False
                    })
                
                predictions = {
                    'boxes': output['boxes'],
                    'labels': output['labels'],
                    'scores': output['scores']
                }
                
                high_score_idxs = predictions['scores'] >= score_threshold
                predictions['boxes'] = predictions['boxes'][high_score_idxs][:max_detections]
                predictions['labels'] = predictions['labels'][high_score_idxs][:max_detections]
                predictions['scores'] = predictions['scores'][high_score_idxs][:max_detections]
                
                matched_preds, unmatched_gts = match_predictions_to_ground_truths(predictions, gt_boxes, iou_threshold)
                
                for pred in matched_preds:
                    
                    cls = pred['pred_label']
                    if pred['gt_box'] is not None:
                        tp[cls] += 1
                    else:
                        fp[cls] += 1
                
                for gt in unmatched_gts:
                    cls = gt['label']
                    fn[cls] += 1
    
    class_metrics = {}
    for cls in range(1, num_classes + 1):
        
        precision = tp[cls] / (tp[cls] + fp[cls] + 1e-6)
        recall = tp[cls] / (tp[cls] + fn[cls] + 1e-6)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-6)
        class_metrics[cls] = {
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1
        }
    
    return class_metrics


def label_accuracy(model, data_loader, device):
    
    correct = 0
    total = 0
    with torch.no_grad():
        
        for images, targets in data_loader:
            
            images = [img.to(device) for img in images]
            
            outputs = model(images)
            
            for i, output in enumerate(outputs):
                
                pred_boxes = output['boxes'].cpu().numpy()
                pred_labels = output['labels'].cpu().numpy()
                gt_boxes = targets[i]['boxes'].cpu().numpy()
                gt_labels = targets[i]['labels'].cpu().numpy()
                
                for gt_label in gt_labels:
                    if gt_label in pred_labels:
                        correct += 1
                
                total += len(gt_labels)
    
    accuracy = correct / total * 100
    
    return accuracy


def detection_accuracy(model, data_loader, device, iou_threshold=0.1, score_threshold=0.1, max_detections=100):
    
    model.eval()
    
    TP = 0
    FP = 0
    FN = 0
    with torch.no_grad():
        for images, targets in data_loader:
            
            images = list(img.to(device) for img in images)
            
            outputs = model(images)
            for i in range(len(outputs)):
                
                pred_boxes = outputs[i]['boxes'].cpu()
                pred_labels = outputs[i]['labels'].cpu()
                pred_scores = outputs[i]['scores'].cpu()
                
                gt_boxes = targets[i]['boxes'].cpu()
                gt_labels = targets[i]['labels'].cpu()
                
                keep = pred_scores >= score_threshold
                pred_boxes = pred_boxes[keep][:max_detections]
                pred_labels = pred_labels[keep][:max_detections]
                pred_scores = pred_scores[keep][:max_detections]
                
                # no predictions, all are FN
                if pred_boxes.size(0) == 0:
                    FN += len(gt_boxes)
                    continue
                
                # no ground truths, all pred are FP
                if gt_boxes.size(0) == 0:
                    FP += len(pred_boxes)
                    continue
                
                # IoU between predicted and ground truth
                ious = box_iou(pred_boxes, gt_boxes)
                
                # each prediction,best matching ground truth
                matched_gt = set()
                for pred_idx in range(ious.size(0)):
                    
                    gt_idx = ious[pred_idx].argmax()
                    max_iou = ious[pred_idx][gt_idx].item()
                    if max_iou >= iou_threshold and gt_labels[gt_idx] == pred_labels[pred_idx]:
                        if gt_idx not in matched_gt:
                            TP += 1
                            matched_gt.add(gt_idx)
                        else:
                            
                            FP += 1
                
                FN += len(gt_boxes) - len(matched_gt)
    
    accuracy = TP / (TP + FP + FN + 1e-6) * 100
    
    return accuracy


def accuracy_(model, data_loader, device, iou_threshold=0.5, score_threshold=0.05, max_detections=100):
    
    model.eval()
    
    corrected = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(tqdm(data_loader, desc="Evaluating Metrics")):
            
            images = list(img.to(device) for img in images)
            outputs = model(images)
            for i, output in enumerate(outputs):
                
                #image_id = targets[i]['image_id'].item()
                
                gt_boxes = targets[i]['boxes'].cpu().numpy()
                gt_labels = targets[i]['labels'].cpu().numpy()
                
                ground_truths = []
                for gt_box, gt_label in zip(gt_boxes, gt_labels):
                    ground_truths.append({
                        'bbox': gt_box,
                        'label': gt_label
                    })
                
                predictions = {
                    'boxes': output['boxes'],
                    'labels': output['labels'],
                    'scores': output['scores']
                }
                
                high_score_idxs = predictions['scores'] >= score_threshold
                predictions['boxes'] = predictions['boxes'][high_score_idxs][:max_detections]
                predictions['labels'] = predictions['labels'][high_score_idxs][:max_detections]
                predictions['scores'] = predictions['scores'][high_score_idxs][:max_detections]
                
                matched_preds, unmatched_gts = match_predictions_to_ground_truths(predictions, ground_truths, iou_threshold)
                
                corrected += len(matched_preds)
                total += len(matched_preds)
                total += len(unmatched_gts)
    
    return corrected / total * 100


#cloner174