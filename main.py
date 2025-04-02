import os
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
import logging
import hydra
from omegaconf import OmegaConf, DictConfig
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Import your dataset and models. Adjust paths as needed.
from utils.PPIDataset import PPIDataset
from utils.model import MLP, OnehotMLP, PositionalMLP  # Choose your model accordingly

class Trainer:
    def __init__(self, cfg: DictConfig):
        """
        Initialize the Trainer with the provided configuration.
        
        Args:
            cfg: Configuration dictionary loaded via Hydra/OmegaConf. It must include 
                 keys for 'data', 'train', 'test', 'model', 'wandb', and 'general'.
        """
        self.cfg = cfg
        self.setup_logging()
        self.load_device()
        
        # Create directory for saving the model checkpoint
        self.model_save_path = cfg.model.model_save_path
        os.makedirs(os.path.dirname(self.model_save_path), exist_ok=True)
        
        # Load the dataset (assumed to be pre-padded with mask information)
        print("Loading dataset...")
        self.load_data()
        # Initialize the model
        print("Loading model...")
        self.load_model()
        
        # Define loss function and optimizer
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=cfg.train.learning_rate)
    
    def setup_logging(self):
        """Set up basic logging to console."""
        logging.basicConfig(
            level=logging.INFO,
            format='[%(asctime)s] [%(levelname)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        self.logger = logging.getLogger(__name__)
    
    def load_device(self):
        """Set device to GPU if available (with seeding), else CPU."""
        cfg = self.cfg
        if torch.cuda.is_available():
            self.device = torch.device(f"cuda:{cfg.general.gpu_id}")
            torch.manual_seed(cfg.general.seed)
            torch.cuda.manual_seed_all(cfg.general.seed)
            self.logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            self.device = torch.device("cpu")
            self.logger.info("No GPU detected, using CPU")
    
    def load_data(self):
        """
        Load the dataset and split into train, validation, and test sets 
        using scikit-learn's train_test_split.
        """
        dataset = PPIDataset(self.cfg.data.json_path)
        total_indices = list(range(len(dataset)))
        cfg_data = self.cfg.data
        
        # Calculate test ratio based on train_split and val_split provided in the config.
        test_ratio = 1 - (cfg_data.train_split + cfg_data.val_split)
        # First, split into train+val and test sets.
        train_val_indices, test_indices = train_test_split(
            total_indices, test_size=test_ratio, random_state=self.cfg.general.seed)
        # Next, split train_val_indices into train and validation sets.
        # The ratio for validation is adjusted relative to the train+val set.
        val_ratio_adjusted = cfg_data.val_split / (cfg_data.train_split + cfg_data.val_split)
        train_indices, val_indices = train_test_split(
            train_val_indices, test_size=val_ratio_adjusted, random_state=self.cfg.general.seed)
        
        self.train_dataset = Subset(dataset, train_indices)
        self.val_dataset = Subset(dataset, val_indices)
        self.test_dataset = Subset(dataset, test_indices)
        
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.cfg.model.batch_size, shuffle=True)
        self.val_loader = DataLoader(self.val_dataset, batch_size=self.cfg.model.batch_size, shuffle=False)
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.cfg.model.batch_size, shuffle=False)
        
        self.logger.info(f"Data loaded: {len(self.train_dataset)} train, "
                         f"{len(self.val_dataset)} val, {len(self.test_dataset)} test samples")
    
    def load_model(self):
        """Initialize the model based on the configuration."""
        cfg_model = self.cfg.model
        # Example: choose OnehotMaskedMLP if baseline is 'onehot', otherwise use MLP.
        if cfg_model.model_choice == "onehot":
            self.model = OnehotMLP(cfg_model).to(self.device)
        elif self.cfg.model.model_choice == 'positional':
            self.model = PositionalMLP(cfg_model).to(self.device)
        else:
            self.model = MLP(cfg_model).to(self.device)
            self.cfg.model.baseline = False
        self.logger.info(f"Model loaded: {cfg_model.model_choice}")
    
    def load_wandb(self):
        """Initialize wandb logging."""
        cfg = self.cfg
        run_id = cfg.wandb.run_id if cfg.wandb.run_id else wandb.util.generate_id()
        OmegaConf.set_struct(cfg, False)
        cfg.wandb.run_id = run_id
        OmegaConf.set_struct(cfg, True)
        self.wandb_run = wandb.init(
            project=cfg.wandb.project,
            name=cfg.wandb.run_name,
            config=OmegaConf.to_container(cfg, resolve=True),
            id=run_id,
            resume=True
        )
        self.logger.info(f"Initialized wandb with run_id: {run_id}")


    def train(self):
        """
        Train the model using the training set and evaluate on the validation set 
        after each epoch. Implements early stopping based on validation accuracy.
        Metrics are logged to wandb.
        """
        self.load_wandb()
        num_epochs = self.cfg.train.num_epochs
        best_val_loss = float('inf')
        patience_counter = 0
        early_stop_patience = self.cfg.train.early_stop_patience

        for epoch in range(num_epochs):
            self.model.train()
            running_loss = 0.0
            correct_train = 0
            total_train = 0
            
            # Training loop with tqdm progress bar
            train_bar = tqdm(self.train_loader, desc=f"Epoch [{epoch+1}/{num_epochs}]", leave=True)

            for embeddings, masks, labels in train_bar:
                embeddings = embeddings.to(self.device)
                masks = masks.to(self.device)
                labels = labels.to(self.device)

                self.optimizer.zero_grad()
                if self.cfg.model.baseline:
                    outputs = self.model(embeddings, masks).squeeze(dim=1)
                else:
                    outputs = self.model(embeddings).squeeze(dim=1)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item() * embeddings.size(0)

                preds = torch.sigmoid(outputs)
                predicted = (preds >= 0.5).float()
                correct_train += (predicted == labels).sum().item()
                total_train += labels.size(0)

                train_acc = correct_train / total_train
                train_bar.set_postfix(loss=f"{running_loss/total_train:.4f}", accuracy=f"{train_acc:.4f}")
            
            train_loss = running_loss / total_train
            val_loss, val_acc = self._evaluate(self.val_loader)

            wandb.log({
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "val_loss": val_loss,
                "val_accuracy": val_acc
            })

            self.logger.info(f"Epoch [{epoch+1}/{num_epochs}] Train Loss: {train_loss:.4f} | "
                             f"Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
            
            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({"model_state_dict": self.model.state_dict()}, self.model_save_path)
                patience_counter = 0
            else:
                patience_counter += 1
                print("Current loss:", val_loss, "Lowest loss:", best_val_loss)
                self.logger.info(f"No improvement in val accuracy for {patience_counter} epoch(s).")
            
            if patience_counter >= early_stop_patience:
                self.logger.info(f"Early stopping triggered at epoch {epoch+1}.")
                break

        wandb.finish()

    
    def _evaluate(self, loader):
        """
        Evaluate the model on the provided DataLoader.
        
        Args:
            loader: DataLoader for the evaluation set.
            
        Returns:
            Tuple of (average_loss, accuracy)
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for embeddings, masks, labels in loader:
                embeddings = embeddings.to(self.device)
                masks = masks.to(self.device)
                labels = labels.to(self.device)
                if self.cfg.model.baseline:
                    outputs = self.model(embeddings, masks).squeeze(dim=1)
                else:
                    outputs = self.model(embeddings).squeeze(dim=1)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item() * embeddings.size(0)
                preds = torch.sigmoid(outputs)
                predicted = (preds >= 0.5).float()
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
        
        avg_loss = total_loss / len(loader.dataset)
        accuracy = correct / total
        return avg_loss, accuracy
    
    def evaluate(self):
        """
        Evaluate the best saved model on the test set.
        """
        checkpoint = torch.load(self.model_save_path)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.logger.info("Evaluating model on test set...")
        
        test_loss, test_acc = self._evaluate(self.test_loader)
        self.logger.info(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")

@hydra.main(version_base=None, config_path="./", config_name="config.yaml")
def main(cfg: DictConfig) -> None:
    # Convert configuration to a mutable container.
    mutable_cfg = OmegaConf.to_container(cfg, resolve=True)
    mutable_cfg = OmegaConf.create(mutable_cfg)

    trainer = Trainer(mutable_cfg)
    if cfg.general.usage == "train":
        trainer.train()
        trainer.evaluate()  # Evaluate on test set after training.
    elif cfg.general.usage == "eval":
        trainer.evaluate()
    elif cfg.general.usage == "feat_extract":
        # Implement feature extraction if needed.
        pass

if __name__ == "__main__":
    main()
    print("\n=============== No Bug No Error, Finished!!! ===============\n")
