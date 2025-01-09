# In the name of God
#
import random
import matplotlib.pyplot as plt
import torch
import numpy as np
import matplotlib.patches as patches


def predict(model, dataloader, device, dataset, num_predictions=5, score_threshold=0.5, is_normal_or_float = 'float', print_label = False):
    """
    is_normal_or_float = [ float  or  normal  or  both  or  none ]
    """
    
    model.eval()
    images_so_far = 0
    fig = plt.figure(figsize=(15, 15))
    idx_to_class = dataset.idx_to_class
    with torch.no_grad():
        
        for batch_idx, (images, targets) in enumerate(dataloader):
            images = list(img.to(device) for img in images)
            outputs = model(images)
            for i, output in enumerate(outputs):
                if images_so_far >= num_predictions:
                    plt.show()
                    return
                
                img = images[i].cpu().permute(1, 2, 0).numpy()
                if is_normal_or_float == 'float' :
                    img = np.array(img * 255 , dtype=np.int64)
                elif is_normal_or_float == 'normal' :
                    img = np.clip(img * np.array([0.229, 0.224, 0.225]) +
                                np.array([0.485, 0.456, 0.406]), 0, 1)
                elif is_normal_or_float == 'both' :
                    img = np.array(img * 255 , dtype=np.int64)
                    img = np.clip(img * np.array([0.229, 0.224, 0.225]) +
                                np.array([0.485, 0.456, 0.406]), 0, 1)
                elif is_normal_or_float == 'none' :
                    pass
                else:
                    raise ValueError("is_normal_or_float Should be one of -> [ float  or  normal  or  both  or  none ]")
                
                boxes = output['boxes'].cpu().numpy()
                labels = output['labels'].cpu().numpy()
                scores = output['scores'].cpu().numpy()

                keep = scores >= score_threshold
                boxes = boxes[keep]
                labels = labels[keep]
                scores = scores[keep]

                ax = plt.subplot(num_predictions // 3 + 1, 3, images_so_far + 1)
                ax.imshow(img)
                ax.axis('off')

                for box, label, score in zip(boxes, labels, scores):
                    xmin, ymin, xmax, ymax = box
                    width, height = xmax - xmin, ymax - ymin

                    rect = patches.Rectangle((xmin, ymin), width, height, linewidth=2, edgecolor='r', facecolor='none')
                    ax.add_patch(rect)
                    class_name = idx_to_class.get(label, 'N/A')
                    if print_label:
                        ax.text(xmin, ymin - 10, f"{class_name}: {score:.2f}",
                                fontsize=12, color='yellow', backgroundcolor='black')

                images_so_far += 1
    plt.show()
    

def predict_random(model, dataloader, device, dataset, num_predictions=5, score_threshold=0.5, is_normal_or_float = 'float', print_label = False):
    """
    is_normal_or_float = [ float  or  normal  or  both  or  none ]
    """
    
    model.eval()
    
    total_images = len(dataloader.dataset)
    random_indices = random.sample(range(total_images), num_predictions)
    fig = plt.figure(figsize=(15, 15))
    idx_to_class = dataset.idx_to_class
    
    for idx in random_indices:
        image, target = dataset[idx]
        image = image.to(device).unsqueeze(0)
        output = model(image)[0]
        img = image.cpu().squeeze().permute(1, 2, 0).numpy()
        if is_normal_or_float == 'float' :
                    img = np.array(img * 255 , dtype=np.int64)
        elif is_normal_or_float == 'normal' :
                    img = np.clip(img * np.array([0.229, 0.224, 0.225]) +
                                np.array([0.485, 0.456, 0.406]), 0, 1)
        elif is_normal_or_float == 'both' :
                    img = np.array(img * 255 , dtype=np.int64)
                    img = np.clip(img * np.array([0.229, 0.224, 0.225]) +
                                np.array([0.485, 0.456, 0.406]), 0, 1)
        elif is_normal_or_float == 'none' :
                    pass
        else:
                    raise ValueError("is_normal_or_float Should be one of -> [ float  or  normal  or  both  or  none ]")
        
        boxes = output['boxes'].cpu().detach().numpy()
        labels = output['labels'].cpu().detach().numpy()
        scores = output['scores'].cpu().detach().numpy()
        
        keep = scores >= score_threshold
        boxes = boxes[keep]
        labels = labels[keep]
        scores = scores[keep]
        
        ax = plt.subplot(num_predictions // 3 + 1, 3, len(fig.axes) + 1)
        ax.imshow(img)
        ax.axis('off')
        for box, label, score in zip(boxes, labels, scores):
            xmin, ymin, xmax, ymax = box
            width, height = xmax - xmin, ymax - ymin
            
            rect = patches.Rectangle((xmin, ymin), width, height, linewidth=2, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            
            class_name = idx_to_class.get(label, 'N/A')
            if print_label:
                ax.text(xmin, ymin - 10, f"{class_name}: {score:.2f}",
                        fontsize=12, color='yellow', backgroundcolor='black')
            #ax.save()
    
    plt.show()