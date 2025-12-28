import os
import sys
import yaml
from pathlib import Path

def verify_yolo_dataset(data_yaml_path):    
    with open(data_yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    
    dataset_path = data['path']
    if not os.path.exists(dataset_path):
        print(f'Cannot find dataset path: {dataset_path}')
        return
    
    print("=" * 50)
    print("Dataset Verification")
    print("=" * 50)
    
    for split in ['train', 'val', 'test']:
        img_dir = os.path.join(dataset_path, 'images', split)
        label_dir = os.path.join(dataset_path, 'labels', split)
        
        if not os.path.exists(img_dir):
            print(f"{split} images directory not found")
            continue
            
        images = list(Path(img_dir).glob('*.jpg')) + list(Path(img_dir).glob('*.png'))
        labels = list(Path(label_dir).glob('*.txt'))
        
        print(f"\n{split.upper()} Split:")
        print(f"  Images: {len(images)}")
        print(f"  Labels: {len(labels)}")
        
        # Check if all images have labels
        missing_labels = 0
        for img in images:
            label_file = os.path.join(label_dir, img.stem + '.txt')
            if not os.path.exists(label_file):
                missing_labels += 1
        
        if missing_labels > 0:
            print(f"Warning: {missing_labels} images without labels")
        else:
            print(f"All images have corresponding labels")
    
    print(f"\nClasses: {data['nc']}")
    print(f"Class names: ", end="")
    for c in list(data["names"].values()):
        print(c, end=', ')

# Run verification
if __name__ == "__main__":
    path = sys.argv[1]
    verify_yolo_dataset(path)