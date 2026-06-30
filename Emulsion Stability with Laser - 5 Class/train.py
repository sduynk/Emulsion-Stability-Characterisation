# train.py
import os
import copy
import torch
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from utils import set_seed, filter_params
from models import init_model
from dataloaders import cross_val_solubility
from sklearn.metrics import confusion_matrix
import torch.nn.functional as F


def idx_to_class(idx, num_classes=5):
    return ["separated", "stable", "separationstarted", "possibleseparated", "notapplicable"][idx]

def metrics(gt, preds, num_classes):

    gt = [idx_to_class(g, num_classes) for g in gt]
    preds = [idx_to_class(p, num_classes) for p in preds]

    accuracy = accuracy_score(gt, preds)
    precision = precision_score(gt, preds, average='macro')
    recall = recall_score(gt, preds, average='macro')
    f1 = f1_score(gt, preds, average='macro')

    accuracy = round(accuracy, 4)
    precision = round(precision, 4)
    recall = round(recall, 4)
    f1 = round(f1, 4)

    return {f"accuracy@{num_classes}": accuracy, f"precision@{num_classes}": precision, f"recall@{num_classes}": recall, f"F1@{num_classes}": f1}


def train_epoch(model, dataloader, optimizer, scheduler, device):
    model.train()
    running_loss = 0.0

    for inputs, labels, paths in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = torch.nn.functional.cross_entropy(outputs, labels, label_smoothing=0.1)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        scheduler.step()

    epoch_loss = running_loss / len(dataloader)
    return epoch_loss


def compute_confusion_matrix(dataloader, model, device, num_classes=5): #seda
    model.eval()
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for inputs, labels, paths in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)

            all_labels.extend(labels.detach().cpu().tolist())
            all_preds.extend(preds.detach().cpu().tolist())

    cm = confusion_matrix(all_labels, all_preds, labels=list(range(num_classes)))
    return cm #seda


def test_model(dataloader, model, device):
    model.eval()
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for inputs, labels, paths in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    metric_dictionary = metrics(all_labels, all_preds, 5)
    return metric_dictionary


def train_model(train_dataloader, val_dataloader, model, optimizer, scheduler, device, num_epochs=30):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_f1 = -1.0

    for epoch in range(num_epochs):
        print(f"Epoch {epoch}/{num_epochs - 1}")
        print("-" * 10)

        train_loss = train_epoch(model, train_dataloader, optimizer, scheduler, device)
        print(f"Train Loss: {train_loss:.4f}")

        val_metrics = test_model(val_dataloader, model, device)
        print(f"VAL Metrics: {val_metrics}")

        if val_metrics["F1@5"] > best_f1:
            best_f1 = val_metrics["F1@5"]
            best_model_wts = copy.deepcopy(model.state_dict())

    print(f"Best VAL F1 Score: {best_f1:.4f}")
    model.load_state_dict(best_model_wts)
    return model


def train_cross_val(config, save_path=None):
    test_results = []
    val_results = []

    for fold, (train_loader, val_loader, test_loader) in enumerate(cross_val_solubility(config)):
        set_seed(config["seed"])  # seed for reproducibility

        model = init_model(config).to(config['device'])
        params = filter_params(model) # remove weight decay from bias and batchnorm layers
        optimizer = torch.optim.AdamW([
                {'params': params['decay'], 'weight_decay': config['weight_decay']},
                {'params': params['no_decay'], 'weight_decay': 0.0}],
                lr=config['lr'])

        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=config['lr'],
            steps_per_epoch=len(train_loader),
            epochs=config['num_epochs'],
            pct_start=0.1
        )
        
        device = config['device']
        num_epochs = config['num_epochs']

        # Train
        train_model(train_loader, val_loader, model, optimizer, scheduler, device, num_epochs)

        # Final eval on val + test
        val_metrics = test_model(val_loader, model, device)
        test_metrics = test_model(test_loader, model, device)

        test_cm = compute_confusion_matrix(test_loader, model, device, num_classes=5)
        print(f"\n[Fold {fold}] TEST confusion matrix (rows=true, cols=pred):\n{test_cm}\n")

        # Misclassified samples
        misclassified_df = collect_misclassified(test_loader, model, device, num_classes=5)
        print(misclassified_df.head(20))

        val_results.append(val_metrics)
        test_results.append(test_metrics)

        if save_path:
            os.makedirs(save_path, exist_ok=True)
            print(f"Saving model for fold {fold}...")

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "test_metrics": test_metrics,
                    "test_cm": test_cm,
                },
                os.path.join(save_path, f"fold_{fold}.pth"),
            )

            misclassified_df.to_csv(
                os.path.join(save_path, f"fold_{fold}_misclassified.csv"),
                index=False
            )

        print(f"[Fold {fold}] TEST Metrics: {test_metrics}")

    val_results = pd.DataFrame(val_results)
    test_results = pd.DataFrame(test_results)

    return val_results, test_results



def train_no_cv(config, save_path=None):

    # Some hacky code just to train a single model without cross-validation
    # Initially cv was to be used for everything, but this was very costly for hparam optimization
    # Ultimately a decision was made that it would be better to explore more hparams over a single train, val split
    # than to explore fewer params over multiple splits given the time constraints
    
    set_seed(config['seed']) # seed for reproducibility

    for fold, (train_loader, val_loader, test_loader) in enumerate(cross_val_solubility(config)):

        model = init_model(config).to(config['device'])
        params = filter_params(model) # remove weight decay from bias and batchnorm layers
        optimizer = torch.optim.AdamW([
                {'params': params['decay'], 'weight_decay': config['weight_decay']},
                {'params': params['no_decay'], 'weight_decay': 0.0}],
                lr=config['lr'])
        
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=config['lr'], steps_per_epoch=len(train_loader), epochs=config['num_epochs'], pct_start=0.1)
        
        device = config['device']
        num_epochs = config['num_epochs']

        train_model(train_loader, val_loader, model, optimizer, scheduler, device, num_epochs)
        val_metrics = test_model(val_loader, model, device)
        test_metrics = test_model(test_loader, model, device)

        test_cm = compute_confusion_matrix(test_loader, model, device, num_classes=5)
        print(f"\n[NO-CV] TEST confusion matrix:\n{test_cm}\n")

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True) if os.path.dirname(save_path) else None
            print("Saving model...")
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "test_metrics": test_metrics,
                    "test_cm": test_cm,
                },
                f"{save_path}.pth",
            )

        print(f"[NO-CV] TEST Metrics: {test_metrics}")
        break

    return val_metrics, test_metrics



def collect_misclassified(dataloader, model, device, num_classes=5):
    model.eval()
    rows = []

    with torch.no_grad():
        for inputs, labels, paths in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            probs = F.softmax(outputs, dim=1)

            preds = torch.argmax(outputs, dim=1)
            confs = torch.max(probs, dim=1).values

            labels_cpu = labels.detach().cpu().tolist()
            preds_cpu = preds.detach().cpu().tolist()
            confs_cpu = confs.detach().cpu().tolist()

            for path, true_idx, pred_idx, conf in zip(paths, labels_cpu, preds_cpu, confs_cpu):
                if true_idx != pred_idx:
                    rows.append({
                        "file": path,
                        "true_label": idx_to_class(true_idx, num_classes),
                        "pred_label": idx_to_class(pred_idx, num_classes),
                        "pred_confidence": round(conf, 4)
                    })

    return pd.DataFrame(rows)