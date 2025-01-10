# in the name of God
#
import os
import torch
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
import numpy as np
import pandas as pd



def convert_bbox_format(boxes, image_size):
    
    img_width, img_height = image_size
    
    converted_boxes = []
    for box in boxes:
        
        x_center, y_center, width, height = box
        
        xmin = (x_center - width / 2) * img_width
        ymin = (y_center - height / 2) * img_height
        xmax = (x_center + width / 2) * img_width
        ymax = (y_center + height / 2) * img_height
        
        xmin = max(0, xmin)
        ymin = max(0, ymin)
        xmax = min(img_width, xmax)
        ymax = min(img_height, ymax)
        
        converted_boxes.append([xmin, ymin, xmax, ymax])
    
    return converted_boxes



class TraficDataset(Dataset):
    
    def __init__(self, annotations_file, img_dir, transform=None, class_to_idx=None):
        
        self.img_dir = img_dir
        self.img_labels = pd.read_csv(annotations_file)
        self.transform = transform
        self.img_labels_grouped = self.img_labels.groupby('image_name').groups
        self.image_ids = {image_name: idx for idx, image_name in enumerate(self.img_labels_grouped.keys())}
        
        if class_to_idx is None:
            classes = sorted(self.img_labels['class'].unique())
            self.class_to_idx = {cls_name: idx + 1 for idx, cls_name in enumerate(classes)}
            self.class_to_idx['__background__'] = 0
        else:
            self.class_to_idx = class_to_idx
        
        self.idx_to_class = {idx: cls_name for cls_name, idx in self.class_to_idx.items()}
    
    
    def __len__(self):
        
        return len(self.img_labels_grouped)
    
    
    def __getitem__(self, idx):
        
        image_name = list(self.img_labels_grouped.keys())[idx]
        group = self.img_labels.iloc[self.img_labels_grouped[image_name]]
        img_path = os.path.join(self.img_dir, image_name)
        image = Image.open(img_path).convert("RGB")
        image_np = np.array(image)
        image_height, image_width, _ = image_np.shape
        boxes = []
        labels = []
        
        for _, row in group.iterrows():
            # x_center, y_center, width, height
            box = [
                float(row['xmin']),
                float(row['ymin']),
                float(row['xmax']),
                float(row['ymax'])
            ]
            
            # xmin, ymin, xmax, ymax
            converted_box = convert_bbox_format([box], (image_width, image_height))[0]
            boxes.append(converted_box)
            class_name = row['class']
            labels.append(self.class_to_idx.get(class_name, 0))
        
        if self.transform:
            transformed = self.transform(image=image_np, bboxes=boxes, labels=labels)
            image_np = transformed['image']
            boxes = transformed['bboxes']
            labels = transformed['labels']
        
        else:
            image_np = transforms.ToTensor()(image)
        
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        
        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([self.image_ids[image_name]], dtype=torch.int64)
        }
        
        return image_np, target
    

#cloner174