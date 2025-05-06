# Data manipulation and visualization
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import cv2
import imagehash

# PyTorch and related libraries
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from torchvision.transforms import InterpolationMode

# PyTorch Lightning
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    Timer,
)
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
from pytorch_lightning.tuner.tuning import Tuner

# Metrics
from torchmetrics.regression import R2Score
from torchmetrics.classification import BinaryAccuracy

# Joblib for saving/loading models
import joblib

# Additional utilities
import time
from typing import List, Optional, Any
import random
from collections import deque
from IPython.display import display as dp
from helper import *

# Lime
from lime import lime_image
from skimage.segmentation import mark_boundaries



def detect_blurry_images(full_df: pd.DataFrame, blur_threshold: float) -> List[str]:
    """
    Detect blurry images based on the variance of the Laplacian.

    Args:
        full_df (pd.DataFrame): DataFrame containing image file names.
        blur_threshold (float): Variance threshold below which images are considered blurry.

    Returns:
        List[str]: List of file names of blurry images.
    """
    blur_ls = []

    file_name_ls = full_df.file_name.tolist()

    for file_name in file_name_ls:

        img = cv2.imread(file_name)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Calculate the Laplacian and then the variance
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

        if laplacian_var < blur_threshold:
            blur_ls.append(file_name)

    return blur_ls

def detect_broken_images(file_name_ls: List[str]) -> List[str]:
    """
    Detect broken images by verifying their integrity and accessibility.

    Args:
        file_name_ls (List[str]): List of image file names to check.

    Returns:
        List[str]: List of file names of broken images.
    """
    broken_img_ls = []

    for file_name in file_name_ls:
        try:
            # verify image integrity
            with Image.open(file_name) as img:
                img.verify()
            # check if image is loadable and accessible
            with Image.open(file_name) as img:
                img.getdata()

        except Exception as e:
            broken_img_ls.append(file_name)
            print(f"Image {file_name} is broken: {e}")

    return broken_img_ls


def compare_img_hash(file_name_ls: List[str], dup_keep: Optional[bool] = False) -> Optional[pd.DataFrame]:
    """
    Compare images by their hash values to find duplicates.

    Args:
        file_name_ls (List[str]): List of image file names to compare.
        dup_keep (Optional[bool]): Whether to keep duplicates in the result.

    Returns:
        Optional[pd.DataFrame]: DataFrame of duplicate images if found, otherwise None.
    """    
    img_hash_ls = []

    for file_name in file_name_ls:
        with Image.open(file_name) as img:
            img_hash = imagehash.dhash(img)

            img_hash_ls.append(str(img_hash))

    file_img_hash_df = pd.DataFrame(
        {"file_name": file_name_ls, "img_hash": img_hash_ls}
    )

    duplicated_hash = file_img_hash_df["img_hash"].duplicated(
        keep=dup_keep
    )  # keep the duplicated pairs for visual comparison

    if sum(duplicated_hash) > 0:
        return file_img_hash_df[duplicated_hash].sort_values(by="img_hash")

    else:
        print("No duplicated image hashes found")

class SqueezenetDataset(Dataset):
    """
    A PyTorch Dataset for loading images and labels for SqueezeNet model training or evaluation.

    Args:
        df (pd.DataFrame): DataFrame containing file names and labels.
        set_type (str): Type of dataset ('train' or 'test') to determine transformations.
    """    
    def __init__(self, df, set_type="train"):
        self.df = df
        self.file_path_ls = df.file_name.tolist()
        self.label_ls = df.label_tensor.tolist()

        # only apply augmentation to train set
        if set_type == "train":
            self.transform = transforms.Compose(
                [
                    transforms.Resize(
                        size=[256], interpolation=InterpolationMode.BILINEAR
                    ),
                    transforms.CenterCrop(size=[224]),
                    transforms.RandomHorizontalFlip(),  # augmentation
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )
        else:
            self.transform = transforms.Compose(
                [
                    transforms.Resize(
                        size=[256], interpolation=InterpolationMode.BILINEAR
                    ),
                    transforms.CenterCrop(size=[224]),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        x = self.file_path_ls[idx]
        y = self.label_ls[idx]

        with Image.open(x) as img:
            x = self.transform(img)

        return x, y

class SqueezenetDataLoader(pl.LightningDataModule):
    """
    A PyTorch Lightning DataModule for loading SqueezeNet datasets.

    Args:
        batch_size (int): Size of batches for training and validation.
    """

    def __init__(self, batch_size: int = 32):
        super().__init__()
        self.batch_size = batch_size

    def setup(self, stage=None):
        train_df = joblib.load('data/train_test_data/train_df.pkl')
        val_df = joblib.load('data/train_test_data/val_df.pkl')
        test_df = joblib.load('data/train_test_data/test_df.pkl')

        self.train_dataset = SqueezenetDataset(train_df, set_type="train")
        self.val_dataset = SqueezenetDataset(val_df, set_type="val")
        self.test_dataset = SqueezenetDataset(test_df, set_type="test")
    
    @staticmethod
    def worker_init_fn(worker_id: int) -> None:
        seed = 0
        np.random.seed(seed + worker_id)
        random.seed(seed + worker_id)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True,
            worker_init_fn=self.worker_init_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True,
            worker_init_fn=self.worker_init_fn,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True,
            worker_init_fn=self.worker_init_fn,
        )

class SqueezenetModel(pl.LightningModule):
    """
    A PyTorch Lightning module for multitask age and gender prediction using SqueezeNet.

    Args:
        is_fine_tune_mode (bool): Indicates whether to fine-tune the model. Default is False.
        lr (float): Learning rate for the optimizer. Default is 0.001.
        class_weights (Optional[torch.Tensor]): Class weights for handling imbalanced classes. Default is None.
        gender_loss_scale (float): Scaling factor for gender loss. Default is 193.0.
    """

    def __init__(
        self,
        is_fine_tune_mode=False,
        lr=0.001,
        class_weights=None,
    ):
        super().__init__()

        # save hyperparameters
        self.save_hyperparameters()

        # parameters
        self.lr = lr
        self.is_fine_tune_mode = is_fine_tune_mode

        # model components
        self.model = models.squeezenet1_1(weights="DEFAULT")

        feature_extractor = nn.Sequential(
            nn.Dropout(0.5),
            nn.Conv2d(512, 1000, kernel_size=(1, 1), stride=(1, 1)),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.model.classifier = feature_extractor

        self.age_pred = nn.Sequential(nn.Linear(1000, 1))
        self.gender_pred = nn.Sequential(nn.Linear(1000, 1))

        # loss
        self.age_criterion = nn.MSELoss()
        self.gender_criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(1.0955, dtype=torch.float32))

        # metrics
        self.gender_acc = BinaryAccuracy()
        self.age_r2_score = R2Score()

    def forward(self, x):
        x = self.model(x)
        age_pred = self.age_pred(x)
        gender_pred = self.gender_pred(x)
        age_gender_pred = torch.cat([age_pred, gender_pred], dim=1)
        return age_gender_pred

    def training_step(self, batch, batch_idx):
        x = batch[0]
        age_true = batch[1][:, 0]
        gender_true = batch[1][:, 1]

        pred = self.forward(x)
        age_pred, gender_pred = pred[:, 0], pred[:, 1]
        age_loss = self.age_criterion(age_pred, age_true)
        gender_loss = self.gender_criterion(gender_pred, gender_true)
        loss = self._calc_total_loss(age_pred, gender_pred, age_true, gender_true)
        
        self.log(name="train_loss", value=loss, on_epoch=True)
        self.log(name='train_age_loss', value=age_loss, on_epoch=True)
        self.log(name='train_gender_loss', value=gender_loss, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):

        x = batch[0]
        age_true = batch[1][:, 0]
        gender_true = batch[1][:, 1]

        pred = self.forward(x)
        age_pred, gender_pred = pred[:, 0], pred[:, 1]
        age_loss = self.age_criterion(age_pred, age_true)
        gender_loss = self.gender_criterion(gender_pred, gender_true)
        loss = self._calc_total_loss(age_pred, gender_pred, age_true, gender_true)

        self.log(name="val_loss", value=loss, on_epoch=True)
        self.log(name='val_age_loss', value=age_loss, on_epoch=True)
        self.log(name='val_gender_loss', value=gender_loss, on_epoch=True)

        # update metrics
        self.age_r2_score.update(age_pred, age_true)  
        self.gender_acc.update(gender_pred, gender_true.int()) 

        return loss

    def on_validation_epoch_end(self, outputs= None) -> None:
        self.log("gender_acc", self.gender_acc.compute(), on_epoch=True)
        self.log('age_r2', self.age_r2_score.compute(), on_epoch=True)
        self.gender_acc.reset()
        self.age_r2_score.reset()

    def predict_step(self, batch, batch_idx):
        x = batch[0]
        pred = self.forward(x)
        age_pred, gender_pred = pred[:, 0], pred[:, 1]

        # convert gender logits to probabilities without tracking gradients
        with torch.no_grad():
            gender_pred = torch.sigmoid(gender_pred)

        return np.array(age_pred.cpu()), np.array(gender_pred.cpu()) 

    def configure_optimizers(self):

        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.lr/40, weight_decay=0.01
        )
        scheduler = torch.optim.lr_scheduler.CyclicLR(
            optimizer,
            base_lr=self.lr/40,
            max_lr=self.lr,
            step_size_up=38,
            mode="triangular",
            cycle_momentum=False,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    def _set_fine_tune_mode(self):
        if self.is_fine_tune_mode:  # fine_tune_mode: True
            for param in self.model.parameters():
                param.requires_grad = True
            print("Fine_tune_mode: True")
        else:  # fine_tune_mode: False
            for name, param in self.model.named_parameters():
                if "features" in name:
                    param.requires_grad = False
            print("Fine_tune_mode: False")

    def _calc_total_loss(self, age_pred, gender_pred, age_true, gender_true):
        age_loss = self.age_criterion(age_pred, age_true)
        gender_loss = self.gender_criterion(gender_pred, gender_true)
        # 193 is an observed value
        scaled_gender_loss = 193 * gender_loss
        return age_loss + scaled_gender_loss

    def change_learning_rate(self, new_lr):
        self.lr = new_lr
        print(f"Update learning rate to {self.lr}")

    def toggle_fine_tune_mode(self, updated_fine_tune_mode, new_lr):
        self.is_fine_tune_mode = updated_fine_tune_mode
        self._set_fine_tune_mode()
        self.lr = new_lr
        print(f"Change learning rate to:{self.lr}")


def initial_lr_finder(model: nn.Module, data_loader: Any, batch_size: int) -> pd.DataFrame:
    """
    Finds an optimal learning rate by training the model with a cyclical learning rate scheduler.

    Args:
        model (nn.Module): The PyTorch model to train.
        data_loader (Any): DataLoader object containing training data.
        batch_size (int): The batch size to use during training.

    Returns:
        pd.DataFrame: DataFrame containing learning rates and corresponding losses.
    """
    pl.seed_everything(0)
    set_seed(0)

    data_loader.setup()
    train_data_loader = data_loader.train_dataloader()
    dataset_len = len(data_loader.train_dataloader().dataset)

    # set optimizer
    optimizer = optim.SGD(model.parameters())
    # set cyclical learning rate scheduler
    step_size_up = int(np.ceil(dataset_len/batch_size))
    
    scheduler = torch.optim.lr_scheduler.CyclicLR(
        optimizer, 
        base_lr=1e-5, 
        max_lr=1e-3, 
        step_size_up=step_size_up, 
        mode='triangular'
        )
    # criterion
    age_criterion = nn.MSELoss()
    gender_criterion = nn.BCEWithLogitsLoss()

    def calc_total_loss(age_pred, gender_pred, age_true, gender_true):
        age_loss = age_criterion(age_pred, age_true)
        gender_loss = gender_criterion(gender_pred, gender_true)
        return 0.5*age_loss + 0.5*gender_loss

    num_epochs = 3
    lr_ls = []
    loss_ls = []

    # loop through batches to collect lr and loss
    for batch_id, batch in enumerate(train_data_loader):
        # forward pass
        outputs = model(batch[0])
        age_true, gender_true = batch[1][:, 0], batch[1][:, 1]
        age_pred, gender_pred = outputs[:, 0], outputs[:, 1]

        loss = calc_total_loss(age_pred, gender_pred, age_true, gender_true)

        # backward pass
        optimizer.zero_grad()
        loss.backward()
        # gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        # update gradient
        optimizer.step()

        # update the scheduler
        scheduler.step()

        # get lr and loss
        current_lr = optimizer.param_groups[0]['lr']
        lr_ls.append(current_lr)
        loss_ls.append(loss.item())

    loss_lr_df = pd.DataFrame(
    {
        'lr': [lr_ls[i] for i in range(0, step_size_up, 2)],
        'loss': moving_average([loss_ls[i] for i in range(0, step_size_up, 2)], 10)
    }
)

    return loss_lr_df
        

class TimeLoggingCallback(pl.Callback):
    """
    A PyTorch Lightning callback that logs the training duration.

    Attributes:
        train_start_time (float): The start time of the training.
    """
    def on_train_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """
        Records the start time of the training.

        Args:
            trainer (pl.Trainer): The PyTorch Lightning Trainer instance.
            pl_module (pl.LightningModule): The PyTorch Lightning model instance.
        """
        self.train_start_time = time.time()

    def on_train_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """
        Calculates and logs the training duration when training ends.

        Args:
            trainer (pl.Trainer): The PyTorch Lightning Trainer instance.
            pl_module (pl.LightningModule): The PyTorch Lightning model instance.
        """
        train_end_time = time.time()
        train_duration = train_end_time - self.train_start_time
        if trainer.logger is not None:
            trainer.logger.log_metrics({"train_duration": train_duration})

def train_model(
    model: pl.LightningModule,
    dataloader: pl.LightningDataModule,
    model_name: str,
    epochs: int,
    model_checkpoint: pl.Callback,
    timer: pl.Callback,
) -> None:
    
    """
    Trains a PyTorch Lightning model with specified configurations and logging.

    Args:
        model (pl.LightningModule): The PyTorch Lightning model to be trained.
        dataloader (pl.LightningDataModule): The PyTorch Lightning DataModule containing the data.
        model_name (str): The name of the model for logging purposes.
        epochs (int): The number of epochs for training.
        model_checkpoint (pl.Callback): A PyTorch Lightning callback for saving model checkpoints.
        timer (pl.Callback): A callback for timing the training process.

    Returns:
        None
    """

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=10,
        mode="min",
        verbose=True,
    )

    lr_monitor = LearningRateMonitor(logging_interval="step")

    time_logging = TimeLoggingCallback()

    callbacks = [early_stopping, lr_monitor, model_checkpoint, time_logging, timer]

    csv_logger = CSVLogger(save_dir="logs/mon/", name=model_name)
    tb_logger = TensorBoardLogger("tb_logs/mon/", name=model_name)

    trainer = pl.Trainer(
        max_epochs=epochs,
        callbacks=callbacks,
        accelerator="gpu",
        logger=[csv_logger, tb_logger],
        precision=16,
        gradient_clip_val=1.0,
        enable_model_summary=False,
        enable_progress_bar=False,
    )

    trainer.fit(model, dataloader)


def train_head_pipeline(
    dataloader: pl.LightningDataModule,
    model: pl.LightningModule,
    model_name: str,
    epoch: int
) -> pl.LightningModule:
    """
    Trains a model using a specified data loader, saves the best model checkpoint based on validation loss,
    and prints the training time.

    Args:
        dataloader (pl.LightningDataModule): The PyTorch Lightning DataModule containing the data.
        model (pl.LightningModule): The PyTorch Lightning model to be trained.
        model_name (str): The name of the model for logging purposes.
        epoch (int): The number of epochs for training.

    Returns:
        pl.LightningModule: The trained PyTorch Lightning model.
    """
    head_train_model_cp = ModelCheckpoint(
        save_top_k=1,
        mode="min",
        monitor="val_loss",
        filename="{epoch:02d}-{val_loss:.2f}",
    )

    timer = Timer()
    train_model(
        model=model,
        dataloader=dataloader,
        model_name=model_name,
        epochs=epoch,
        model_checkpoint=head_train_model_cp,
        timer=timer,
    )
    formatted_time = time.strftime("%M:%S", time.gmtime(timer.time_elapsed("train")))
    print(f"{model_name} head training time ({epoch} epochs):")
    print(formatted_time)

    return model


def fine_tune_pipeline(
    dataloader: pl.LightningDataModule,
    head_trained_model: pl.LightningModule,
    model_name: str,
    lr: float,
    epoch: int
) -> pl.LightningModule:
    """
    Fine-tunes a pre-trained model with specified data loader, saves the best model checkpoint based on validation loss,
    and prints the training time.

    Args:
        dataloader (pl.LightningDataModule): The PyTorch Lightning DataModule containing the data.
        head_trained_model (pl.LightningModule): The pre-trained PyTorch Lightning model to be fine-tuned.
        model_name (str): The name of the model for logging purposes.
        batch_size (int): The batch size for training.
        lr (float): The learning rate for fine-tuning.
        epoch (int): The number of epochs for fine-tuning.

    Returns:
        pl.LightningModule: The fine-tuned PyTorch Lightning model.
    """
    model = head_trained_model
    model.toggle_fine_tune_mode(True, lr)

    model_cp = ModelCheckpoint(
        save_top_k=1,
        mode="min",
        monitor="val_loss",
        filename="{epoch:02d}-{val_loss:.2f}",
    )

    timer = Timer()
    train_model(
        model=model,
        dataloader=dataloader,
        model_name=model_name,
        epochs=epoch,
        model_checkpoint=model_cp,
        timer=timer,
    )

    formatted_time = time.strftime("%M:%S", time.gmtime(timer.time_elapsed("train")))
    print(f"{model_name} head training time ({epoch} epochs):")
    print(formatted_time)

    return model

def prediction_pipeline(
    dataloader: pl.LightningDataModule,
    fine_tuned_model: pl.LightningModule,
    dataset_type: str = "test"
) -> List[Any]:
    """
    Runs a prediction pipeline on a given test dataset using a fine-tuned model.

    Args:
        dataloader (pl.LightningDataModule): The data module containing the test dataset.
        fine_tuned_model (pl.LightningModule): The fine-tuned model for making predictions.

    Returns:
        List[Any]: A list of predictions generated by the model on the test dataset.
    """
    dataloader.setup("test")
    model = fine_tuned_model
    if dataset_type == "val":
        test_dataloader = dataloader.val_dataloader()
    else:
        test_dataloader = dataloader.test_dataloader()
    trainer = pl.Trainer(enable_model_summary=False, enable_progress_bar=False)

    predictions = trainer.predict(model, dataloaders=test_dataloader)

    return predictions

def set_seed(seed: int) -> None:
    """
    Sets the seed for generating random numbers to ensure reproducibility.

    This function sets the seed for Python's `random` module, NumPy, and PyTorch.
    Additionally, it configures PyTorch to use deterministic algorithms to further
    ensure reproducible results. If CUDA is available, the seed is also set for
    CUDA operations.

    Args:
        seed (int): The seed value to use for random number generation.

    Returns:
        None
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def moving_average(data: List[float], window_size: int) -> List[float]:
    """
    Computes the moving average of the given data using a specified window size.

    Args:
        data (List[float]): A list of numerical data to compute the moving average on.
        window_size (int): The size of the window to use for the moving average.

    Returns:
        List[float]: A list of moving averages computed from the input data.
    """
    window = deque(maxlen=window_size)
    moving_averages = []

    for x in data:
        window.append(x)
        moving_averages.append(np.mean(window))
    return moving_averages



def stack_to_tensor(df: pd.DataFrame) -> torch.Tensor:
    """
    Converts a DataFrame into a stacked tensor using the SqueezenetDataset.

    Args:
        df (pd.DataFrame): The input DataFrame containing the data to be converted.

    Returns:
        torch.Tensor: A stacked tensor containing the input data from the DataFrame.
    """
    dataset = list(SqueezenetDataset(df))
    stacked_tensor = torch.stack([i[0] for i in dataset])
    return stacked_tensor


def plot_lime_explanation(
    explanation_ls: List[lime_image.ImageExplanation], 
    group_name: str, 
    label: int = 1
) -> None:
    """
    Plot LIME explanations for a list of images with class boundaries.

    Args:
        explanation_ls (List[lime_image.ImageExplanation]): List of LIME explanations.
        group_name (str): Title group name for the plot.
        label (int, optional): Target class label to visualize. Defaults to 1.
    """
    fig_size(10, 2)
    for idx, i in enumerate(explanation_ls):
        temp, mask = i.get_image_and_mask(
            label=label,
            positive_only=False,
            num_features=10,
            hide_rest=False,
        )
        img_boundry = mark_boundaries(temp, mask)
        plt.subplot(1, 5, idx + 1)
        plt.imshow(img_boundry)
        plt.axis("off")
    
    plt.suptitle(f"{group_name} in Gender Prediction")
    plt.show()