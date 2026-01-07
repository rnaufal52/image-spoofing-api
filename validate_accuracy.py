import os
import cv2
import glob
import sys
from app.services.modelServices import predict, load_model

# Ensure model is loaded
load_model()

def validate_folder(folder_path, expected_label):
    image_paths = glob.glob(os.path.join(folder_path, "*.*"))
    total = 0
    correct = 0
    errors = 0
    
    print(f"Testing {len(image_paths)} images in {folder_path} (Expected: {expected_label})...")
    
    for img_path in image_paths:
        if not img_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
            
        total += 1
        try:
            image = cv2.imread(img_path)
            if image is None:
                print(f"Skipping unreadable: {img_path}")
                continue
                
            result = predict(image)
            
            # FAIL means Spoof (Fake), PASS means Real
            prediction = result.decision  
            
            is_correct = False
            if expected_label == "Fake":
                if prediction == "FAIL":
                    is_correct = True
            elif expected_label == "Real":
                if prediction == "PASS":
                    is_correct = True
            
            if is_correct:
                correct += 1
            else:
                # Analyze False Negatives (Fake labeled as Real)
                if expected_label == "Fake" and prediction == "PASS":
                     print(f"[MISSED FAKE] Score: {result.mean_score:.4f}, Evidence: {result.evidence}, Reason: {result.reason}")

        except Exception:
            pass
            
    accuracy = (correct / total) * 100 if total > 0 else 0
    return total, correct, accuracy

def analyze_scores(folder_path):
    image_paths = glob.glob(os.path.join(folder_path, "*.*"))
    scores = []
    print(f"Analyzing scores for {len(image_paths)} images in {folder_path}...")
    for img_path in image_paths:
        if not img_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
        try:
            image = cv2.imread(img_path)
            if image is None: continue
            
            # Predict manually to get score
            from app.services.modelServices import model, get_face_crop, transform, DEVICE
            import torch
            
            face_crop = get_face_crop(image)
            if face_crop is None: continue
            
            rgb_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
            tensor = transform(rgb_crop).unsqueeze(0).to(DEVICE)
            with torch.inference_mode():
                output = model(tensor)
                spoof_map = output[0] if isinstance(output, (tuple, list)) else output
                mean_score = float(torch.mean(spoof_map).item())
                scores.append(mean_score)
                
        except Exception:
            pass

    if scores:
        import statistics
        print(f"Stats for {folder_path}:")
        print(f"  Min Score: {min(scores):.4f}")
        print(f"  Max Score: {max(scores):.4f}")
        print(f"  Avg Score: {statistics.mean(scores):.4f}")
        
        # Danger Zone Analysis
        danger_count = sum(1 for s in scores if 0.45 <= s <= 0.60)
        print(f"  Danger Zone (0.45-0.60): {danger_count} images ({(danger_count/len(scores))*100:.2f}%)")
    else:
        print("No scores collected.")

def main():
    base_dir = "train"
    fake_dir = os.path.join(base_dir, "fake")
    real_dir = os.path.join(base_dir, "real")
    
    print("=== STARTING VALIDATION ===")
    
    # Test Fake
    f_total, f_correct, f_acc = validate_folder(fake_dir, "Fake")
    print(f"FAKE Results: {f_correct}/{f_total} Correct ({f_acc:.2f}%)")
    
    # Test Real
    r_total, r_correct, r_acc = validate_folder(real_dir, "Real")
    print(f"REAL Results: {r_correct}/{r_total} Correct ({r_acc:.2f}%)")
    
    total_imgs = f_total + r_total
    total_correct = f_correct + r_correct
    overall_acc = (total_correct / total_imgs) * 100 if total_imgs > 0 else 0
    
    print("===========================")
    print(f"OVERALL ACCURACY: {overall_acc:.2f}%")
    print("===========================")

    # Analyze Scores
    analyze_scores(fake_dir)
    analyze_scores(real_dir)

if __name__ == "__main__":
    main()
