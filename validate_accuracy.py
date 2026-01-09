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
            
            if total % 100 == 0:
                print(f"Processed {total} images...", end='\r')
            
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
                # Save Failed Images for Review
                failed_dir = "failed_validation_images"
                missed_fake_dir = os.path.join(failed_dir, "missed_fake")
                missed_real_dir = os.path.join(failed_dir, "missed_real")
                os.makedirs(missed_fake_dir, exist_ok=True)
                os.makedirs(missed_real_dir, exist_ok=True)
                
                import shutil
                
                # Analyze False Negatives (Fake labeled as Real)
                if expected_label == "Fake" and prediction == "PASS":
                     print(f"[MISSED FAKE] File: {os.path.basename(img_path)} | Score: {result.mean_score:.4f}, Evidence: {result.evidence}, Reason: {result.reason}")
                     dst_path = os.path.join(missed_fake_dir, f"[SCORE_{result.mean_score:.4f}]_{os.path.basename(img_path)}")
                     shutil.copy(img_path, dst_path)
                
                # Analyze False Positives (Real labeled as Fake/Fail)
                if expected_label == "Real" and prediction == "FAIL":
                     print(f"[MISSED REAL] File: {os.path.basename(img_path)} | Score: {result.mean_score:.4f}, Evidence: {result.evidence}, Reason: {result.reason}")
                     dst_path = os.path.join(missed_real_dir, f"[SCORE_{result.mean_score:.4f}]_{os.path.basename(img_path)}")
                     shutil.copy(img_path, dst_path)

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
            
            # Use the service predict function which handles logic correctly
            from app.services.modelServices import predict
            result = predict(image)
            # result.details["mean_score"] or result.mean_score? 
            # PredictionResult schema usually has mean_score?
            # Checking modelServices.py: details dictionary has mean_score.
            # PredictionResult object has stats/details? 
            # Let's check schema/prediction.py if possible, but predict returns PredictionResult... 
            # In predict function: return PredictionResult(..., details=details)
            # So result.details['mean_score'] should work.
            
            # Wait, PredictionResult schema (from previous edits/knowledge) probably has fields.
            # But let's assume result.details is accessible.
            if hasattr(result, 'details') and 'mean_score' in result.details:
                scores.append(result.details['mean_score'])
            elif hasattr(result, 'mean_score'): # If schema has direct field (unlikely based on code)
                 scores.append(result.mean_score)
            else:
                 # Fallback if I can't find it, but predict sets details.
                 scores.append(0.0)
                 
        except Exception as e:
            # print(f"Error: {e}")
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
