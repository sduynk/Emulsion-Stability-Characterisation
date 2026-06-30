from train import train_cross_val
import pandas as pd
import os
import torch
import numpy as np

# suppress warnings due from sklearn metrics
# early in training, metrics such as F1 score are poorly defined
# as model does may not predict any positive classes
import warnings
warnings.filterwarnings("ignore")

def run(config):
    out = []
    cms_sum = np.zeros((5,5), dtype=np.int64)  # total CM across all folds+seeds
    cms_count = 0

    for i in range(0, 3):
        config['seed'] = i
        save_path = os.path.join("Trained_Models_5class", f"{config['model']}", f'seed_{i}')
        val_metrics, test_metrics = train_cross_val(config, save_path=save_path)
        out.append(test_metrics)

        # fold checkpoint’lerinden CM oku ve topla
        for fold in range(5):  # n_splits=5 olduğu için
            pth_path = os.path.join(save_path, f"fold_{fold}.pth")
            ckpt = torch.load(pth_path, map_location="cpu")
            if "test_cm" in ckpt:
                cm = ckpt["test_cm"]
                cm = np.array(cm)  # güvenli
                cms_sum += cm
                cms_count += 1

    result_df = pd.concat(out, ignore_index=True)

    mean_results = result_df.mean(numeric_only=True)
    std_results = result_df.std(numeric_only=True)

    summary_df = pd.DataFrame({'mean': mean_results, 'std': std_results}).sort_index()
    print(summary_df)

    os.makedirs("./results", exist_ok=True)
    result_df.to_csv(os.path.join("./results", config['model'] + '_results.csv'), index=False)

    # FINAL CM (tek tane)
    if cms_count > 0:
        cm_df = pd.DataFrame(
            cms_sum,
            index=["true_separated","true_stable","true_separationstarted","true_possibleseparated","true_notapplicable"],
            columns=["pred_separated","pred_stable","pred_separationstarted","pred_possibleseparated","pred_notapplicable"]
        )
        print("\n=== FINAL CONFUSION MATRIX (SUM over seeds+folds) ===")
        print(cm_df)

        cm_df.to_csv(os.path.join("./results", config['model'] + "_final_confusion_matrix.csv"))
    else:
        print("No confusion matrices were found in checkpoints (test_cm missing).")



def main():
    # choose from the following models: resnet18, efficientnet, convnext
    config = {
        "data_dir": r"C:\Users\sedau\Desktop\emulsion_stability - laser 5 class\LASER_fulldataset_allclean_allinonefolder_labelchange(s)_split_crop_merge_groupid216",
        "model": None,
        "lr": 0.0005,
        "batch_size": 128,
        "weight_decay": 0.001,
        "center_crop": (1080, 1370),
        "resize": (224, 224),
        "degrees": 10,
        "translate": (0.1, 0.1),
        "scale_lower": 0.95,
        "scale_upper": 1.0,
        "num_epochs":30, 
        "seed": 0,
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    }

    for model in ["resnet18", "efficientnet", "convnext"]:
        config["model"] = model
        run(config)

if __name__ == "__main__":
    import torch.multiprocessing as mp
    mp.freeze_support()  # windows için güvenli
    main()