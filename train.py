import os
import yaml
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from dataset_module.graph_dataset import ConceptGraphDataset
from protogtx.helper import Trainer, Evaluator, collate
from protogtx.ConceptGraphTransformer import Classifier

def set_seed(seed):
    """Set seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_config(config_path):
    """Load configuration from a YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    args = parser.parse_args()

    # Load configuration
    cfg = load_config(args.config)
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create save directory
    save_dir = cfg['paths']['save_dir']
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Dataset Setup
    print("Loading datasets...")
    class_dict = {'normal': 0, 'luad': 1, 'lscc': 2}
    
    def get_loader(set_path, shuffle=False):
        ids = open(set_path).readlines()
        ds = ConceptGraphDataset(
            root=cfg['paths']['data_path'],
            root2=cfg['paths']['expl_path'],
            ids=ids,
            site=cfg['model']['site'],
            classdict=class_dict
        )
        return DataLoader(
            ds, 
            batch_size=cfg['train']['batch_size'],
            collate_fn=collate,
            shuffle=shuffle,
            pin_memory=True
        )

    loader_train = get_loader(cfg['paths']['train_set'], shuffle=True)
    loader_val = get_loader(cfg['paths']['val_set'], shuffle=False)

    # Model Setup
    print("Initializing model...")
    model = Classifier(
        cfg['model']['n_class'], 
        n_features=cfg['model']['n_features'], 
        expl_w=5.0
    )
    model = nn.DataParallel(model).to(device)

    if cfg['train']['resume']:
        model.load_state_dict(torch.load(cfg['train']['resume']))

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['train']['lr'], weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 100], gamma=0.1)
    
    writer = SummaryWriter(log_dir=os.path.join(save_dir, cfg['model']['task_name']))
    f_log = open(os.path.join(save_dir, f"{cfg['model']['task_name']}.log"), 'w')

    trainer = Trainer(cfg['model']['n_class'])
    evaluator = Evaluator(cfg['model']['n_class'])

    # Training Loop
    best_acc = 0.0
    for epoch in range(cfg['train']['num_epochs']):
        model.train()
        train_loss, total_samples = 0.0, 0
        
        print(f"\nEpoch {epoch+1}/{cfg['train']['num_epochs']}")

        for i, batch in enumerate(loader_train):
            optimizer.zero_grad()
            preds, labels, cls_loss, expl_loss, _ = trainer.train(batch, model, n_features=cfg['model']['n_features'])
            
            loss = cls_loss + expl_loss
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            total_samples += len(labels)
            trainer.metrics.update(labels, preds)

            if (i + 1) % cfg['train']['log_interval'] == 0:
                print(f"[{total_samples}] Loss: {train_loss/total_samples:.3f} | Acc: {trainer.get_scores():.3f}")

        scheduler.step()

        # Validation
        model.eval()
        with torch.no_grad():
            for batch in loader_val:
                preds, labels, _, _, _ = evaluator.eval_test(batch, model, n_features=cfg['model']['n_features'])
                evaluator.metrics.update(labels, preds)

        val_acc = evaluator.get_scores()
        print(f"Validation Acc: {val_acc:.4f}")

        # Save Checkpoint
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), os.path.join(save_dir, f"best_model.pth"))

        # Log to file and Tensorboard
        f_log.write(f"Epoch {epoch+1}: Train Acc {trainer.get_scores():.4f}, Val Acc {val_acc:.4f}\n")
        f_log.flush()
        writer.add_scalar('Acc/Train', trainer.get_scores(), epoch)
        writer.add_scalar('Acc/Val', val_acc, epoch)

        trainer.reset_metrics()
        evaluator.reset_metrics()

    f_log.close()
    writer.close()

if __name__ == "__main__":
    main()