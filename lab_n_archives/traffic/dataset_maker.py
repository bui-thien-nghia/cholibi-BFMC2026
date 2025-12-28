import os
from PIL import Image
import random

def convert_gtsdb_to_yolo(gtsdb_path, output_path, train_split=0.7, val_split=0.2):
    for split in ['train', 'val', 'test']:
        os.makedirs(f'{output_path}/images/{split}', exist_ok=True)
        os.makedirs(f'{output_path}/labels/{split}', exist_ok=True)
    
    gt_file = os.path.join(gtsdb_path, 'gt.txt')
    annotations = {}
    
    print("Reading annotations...")
    with open(gt_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith('Filename') or line.strip() == '':
                continue
            
            parts = line.strip().split(';')
            if len(parts) < 6:
                continue
                
            filename = parts[0]
            x1, y1, x2, y2 = map(int, parts[1:5])
            class_id = int(parts[5])
            
            if filename not in annotations:
                annotations[filename] = []
            
            annotations[filename].append({
                'bbox': [x1, y1, x2, y2],
                'class': class_id
            })
    
    # Get list of all images with annotations
    image_files = list(annotations.keys())
    random.shuffle(image_files)
    
    # Split dataset
    n_total = len(image_files)
    n_train = int(n_total * train_split)
    n_val = int(n_total * val_split)
    
    train_files = image_files[:n_train]
    val_files = image_files[n_train:n_train + n_val]
    test_files = image_files[n_train + n_val:]
    
    splits = {
        'train': train_files,
        'val': val_files,
        'test': test_files
    }
    
    print(f"Dataset split: Train={len(train_files)}, Val={len(val_files)}, Test={len(test_files)}")
    
    # Process each split
    for split_name, file_list in splits.items():
        print(f"\nProcessing {split_name} split...")
        
        for img_file in file_list:
            src_img = os.path.join(gtsdb_path, img_file)
            if not os.path.exists(src_img):
                src_img = src_img.replace('.ppm', '.jpg')
            
            if not os.path.exists(src_img):
                print(f"Warning: Image not found: {img_file}")
                continue
            
            img = Image.open(src_img)
            img_width, img_height = img.size
            
            output_img_name = img_file.replace('.ppm', '.jpg')
            dst_img = os.path.join(output_path, 'images', split_name, output_img_name)
            dst_label = os.path.join(output_path, 'labels', split_name, 
                                    output_img_name.replace('.jpg', '.txt'))
            
            img.convert('RGB').save(dst_img, 'JPEG')
            
            yolo_annotations = []
            for ann in annotations[img_file]:
                x1, y1, x2, y2 = ann['bbox']
                class_id = ann['class']
                
                x_center = ((x1 + x2) / 2) / img_width
                y_center = ((y1 + y2) / 2) / img_height
                width = (x2 - x1) / img_width
                height = (y2 - y1) / img_height
                
                x_center = max(0, min(1, x_center))
                y_center = max(0, min(1, y_center))
                width = max(0, min(1, width))
                height = max(0, min(1, height))
                
                yolo_annotations.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
            
            # Write label file
            with open(dst_label, 'w') as f:
                f.write('\n'.join(yolo_annotations))
    
    print(f"\n✓ Conversion complete! Dataset saved to: {output_path}")
    print(f"  Train images: {len(train_files)}")
    print(f"  Val images: {len(val_files)}")
    print(f"  Test images: {len(test_files)}")

if __name__ == "__main__":
    GTSDB_PATH = r"dataset\\FullIJCNN2013"
    OUTPUT_PATH = r"dataset\\YOLO_FullIJCNN2013"
    
    convert_gtsdb_to_yolo(
        gtsdb_path=GTSDB_PATH,
        output_path=OUTPUT_PATH,
        train_split=0.7,
        val_split=0.2
    )