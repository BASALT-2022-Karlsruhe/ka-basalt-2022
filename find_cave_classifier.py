from datetime import datetime
import os

from tqdm import tqdm
import numpy as np
import torch as th
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import cv2

from utils import create_subfolders
from utils.logs import Logging
from openai_vpt.agent import resize_image, AGENT_RESOLUTION
from impala_based_models import ImpalaBinaryClassifier

LOG_FILE = f"find_cave_classifier_log_{datetime.now().strftime('%Y:%m:%d_%H:%M:%S')}.log"
DEVICE = th.device("cuda" if th.cuda.is_available() else "cpu")
STACK_SIZE = 1


class FindCaveCNN(nn.Module):
    def __init__(self):
        super().__init__()
        features_dim = 2
        n_input_channels = STACK_SIZE
        first_conv_out_channels =  32
        second_conv_out_channels = 64
        third_conv_out_channels =  64
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, first_conv_out_channels, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(first_conv_out_channels, second_conv_out_channels, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(second_conv_out_channels, third_conv_out_channels, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            test_tensor = th.as_tensor(np.zeros((STACK_SIZE, *AGENT_RESOLUTION))[None]).float()
            n_flatten = self.cnn(test_tensor).shape[1]

        self.linear = nn.Linear(n_flatten, features_dim)

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))

    def predict(self, observations):
        return self.forward(observation).argmax(dim=1)
        #return self.forward(observation) > 0.


def preprocessing(img, greyscale=False, norm=False):
    try:
        img = resize_image(img, AGENT_RESOLUTION)
    except Exception as e:
        print(str(e))
    if greyscale:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if norm:
        # scale pixel values to [-1, 1]
        img = 2 * (img / 255.) - 1
    return img


def process_video(video_path, rgb_images=True):
    video_object = cv2.VideoCapture(video_path)
    img_stacks = []
    if rgb_images:
        current_stack = np.empty((STACK_SIZE, *AGENT_RESOLUTION, 3))
    else:
        current_stack = np.empty((STACK_SIZE, *AGENT_RESOLUTION))
    count = 0
    success = True
    while success:
        success, img = video_object.read()
        if  success and img is not None:
            processed_img = preprocessing(img)
            current_stack[count % STACK_SIZE, ...] = processed_img
            count += 1
            if count % STACK_SIZE == 0:
                img_stacks.append(current_stack)
                if rgb_images:
                    current_stack = np.empty((STACK_SIZE, *AGENT_RESOLUTION, 3))
                else:
                    current_stack = np.empty((STACK_SIZE, *AGENT_RESOLUTION))
    return img_stacks


def count_stacks(video_path):
    video_object = cv2.VideoCapture(video_path)
    frame_count = int(video_object.get(cv2.CAP_PROP_FRAME_COUNT))
    return frame_count // STACK_SIZE


def convert_videos_to_stacks(video_dir, stack_dir, label, stack_idx=0, squeeze=True):
    for file in tqdm(os.listdir(video_dir)):
        filename = os.fsdecode(file)
        if filename.endswith(".mp4"):
            img_stacks = process_video(os.path.join(video_dir, filename))
            for img_stack in img_stacks:
                save_path = os.path.join(stack_dir, f"{label}_{stack_idx}.npy")
                if squeeze:
                    img_stack = img_stack.squeeze()
                np.save(save_path, img_stack)
                stack_idx += 1
    return stack_idx


class FindCaveImageDataset(Dataset):
    def __init__(self, stack_dir, num_stacks=None, balance_classes=True):
        avail_stacks = len(os.listdir(stack_dir))
        self.stack_dir = stack_dir

        self.num_stacks = num_stacks or avail_stacks
        if num_stacks < avail_stacks:
            file_idxs = np.random.choice(avail_stacks, num_stacks, replace=False)
        elif num_stacks == avail_stacks:
            file_idxs = range(avail_stacks)

        # register data to be used
        self.stack_files = []
        self.stack_labels = []
        for file_idx in file_idxs:
            for label in [0, 1]:
                filename = f"{label}_{file_idx}.npy"
                filepath = os.path.join(stack_dir, filename)
                if os.path.exists(filepath):
                    self.stack_files.append(filename)
                    self.stack_labels.append(label)
                    break # each stack can only have one label

        assert len(self.stack_files) == len(self.stack_labels) == self.num_stacks

        if balance_classes:
            # Count labels of examples (assumes that both classes are available)
            labels, counts = np.unique(self.stack_labels, return_counts=True)
            idx0 = np.where(labels == 0)[0][0]
            idx1 = np.where(labels == 1)[0][0]

            # Remove absolute difference of examples from dominating class
            difference_01 = counts[idx0] - counts[idx1]
            self.num_stacks -= abs(difference_01)

            # Randomly choose items to delete from registry lists
            if difference_01 > 0: # more 0 labels
                label_0_idxs = np.where(np.array(self.stack_labels) == 0)[0]
                delete_idxs = np.random.choice(label_0_idxs, abs(difference_01), replace=False)               
            elif difference_01 < 0: # more 1 labels
                label_1_idxs = np.where(np.array(self.stack_labels) == 1)[0]
                delete_idxs = np.random.choice(label_1_idxs, abs(difference_01), replace=False)               

            for del_idx in sorted(delete_idxs, reverse=True):
                del self.stack_labels[del_idx], self.stack_files[del_idx]

        self.labels_onehot = th.tensor([[1, 0], [0, 1]])
                    

    def __len__(self):
        return self.num_stacks

    def __getitem__(self, stack_idx):
        np_stack = np.load(os.path.join(self.stack_dir, self.stack_files[stack_idx]))
        stack = th.from_numpy(np_stack).float()
        label = self.labels_onehot[self.stack_labels[stack_idx]].float()
        return stack, label


def create_dataset(video_dir_cave, video_dir_expl, stack_dir):
    if os.path.exists(stack_dir) and len(os.listdir(stack_dir)) > 0:
        Logging.info(f"Dataset already exists at {stack_dir}.")
    else:
        os.makedirs(stack_dir, exist_ok=True)
        stack_idx = convert_videos_to_stacks(video_dir_cave, stack_dir, 1)
        convert_videos_to_stacks(video_dir_expl, stack_dir, 0, stack_idx=stack_idx)


def train(
        stack_dir,
        model_dir,
        model_name="impala",
        data_frac=0.05,
        validation_frac=0.5,
        report_rate=1000
    ):
    os.makedirs(model_dir, exist_ok=True)

    # hyperparameters
    num_epochs = 1
    batch_size = 64
    lr = 0.001

    # only use fraction of the dataset
    if os.path.exists(stack_dir) and len(os.listdir(stack_dir)) > 0:
        num_stacks = len(os.listdir(stack_dir))
    else:
        raise ValueError(f"No data found at location {stack_dir}")
    num_stacks = int(num_stacks * data_frac)

    dataset = FindCaveImageDataset(stack_dir, num_stacks)

    # actual numbers after creating dataset (could be different due to balancing)
    num_validation_stacks = int(dataset.num_stacks * validation_frac)
    num_training_stacks = dataset.num_stacks - num_validation_stacks
    Logging.info(f"#stacks: {dataset.num_stacks} = #training stacks: {num_training_stacks} + #validation stacks: {num_validation_stacks}")

    # split into train and validation sets
    training_dataset, validation_dataset = random_split(
        dataset,
        [num_training_stacks, num_validation_stacks],
        generator=th.Generator().manual_seed(42),
    )

    training_loader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True, num_workers=6)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, num_workers=6)

    # create model, optimizer, loss function
    if model_name.lower() == "impala":
        # TODO parameterize model_width
        model = ImpalaBinaryClassifier().to(DEVICE)
    elif model_name.lower() == "naturecnn":
        model = FindCaveCNN().to(DEVICE)

    optimizer = th.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss() # nn.BCEWithLogitsLoss()

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    best_vloss = 1_000_000.

    for epoch in range(num_epochs):

        Logging.info(f"Epoch {epoch + 1}")

        model.train(True)
        running_loss = last_loss = 0
        last_accuracy = 0.0
        correct = 0
        for i, (stacks, labels) in enumerate(tqdm(training_loader)):
            stacks, labels = stacks.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            # Calculate loss
            logit_pred = model(stacks)#.squeeze()
            loss = loss_fn(logit_pred, labels)

            # Backprop
            loss.backward()
            optimizer.step()
            
            with th.no_grad():
                # Calculate accuracy
                pred = logit_pred.argmax(dim=1).long()#(logit_pred.sigmoid() > 0).long()
                digit_labels = labels.argmax(dim=1).long()
                #print("logit_pred size:", logit_pred.size())
                #print("pred size:", pred.size())
                correct += (pred == digit_labels).float().sum()

            del stacks, labels
            th.cuda.empty_cache()

            running_loss += loss.item()
            if i % report_rate == report_rate - 1:
                last_loss = running_loss / report_rate
                last_accuracy = 100 * correct / (report_rate * batch_size)
                tqdm.write("Batch: {}, Loss: {:.4f}, Accuracy: {:.2f}".format(i + 1, last_loss, last_accuracy))
                running_loss = 0
                correct = 0
        model.train(False)

        with th.no_grad():
            running_vloss = last_accuracy = 0.0
            vcorrect = 0
            for i, (vstacks, vlabels) in enumerate(tqdm(validation_loader)):
                vstacks, vlabels = vstacks.to(DEVICE), vlabels.to(DEVICE)

                # Calculate loss
                vlogit_pred = model(vstacks)#.squeeze()
                #if vlogit_pred.size() == th.Size([]):
                #    vlogit_pred = vlogit_pred.unsqueeze(0)
                vloss = loss_fn(vlogit_pred, vlabels)
                running_vloss += vloss

                # Calculate accuracy
                vpred = vlogit_pred.argmax(dim=1).long() # (vlogit_pred.sigmoid() > 0).long()
                vdigit_labels = vlabels.argmax(dim=1).long()
                vcorrect += (vpred == vdigit_labels).float().sum()

                del vstacks, vlabels
                th.cuda.empty_cache()
            avg_vloss = running_vloss / (i + 1)
            avg_accuracy = 100 * vcorrect / len(validation_dataset)

        Logging.info('Loss: train {:.4f} / valid {:.4f}, Accuracy: train {:.2f} / valid {:.2f}'.format(last_loss, avg_vloss, last_accuracy, avg_accuracy))

        # if avg_vloss < best_vloss: # this only makes sense for many epochs
        best_vloss = avg_vloss
        model_file = 'FindCaveCNN_{}_epoch{}.weights'.format(timestamp, epoch + 1)
        model_path = os.path.join(model_dir, model_file)
        Logging.info(f"Saving model to {model_file}")
        th.save(model.state_dict(), model_path)


if __name__ == "__main__":
    create_subfolders.main()
    Logging.setup(name=LOG_FILE)
    Logging.info("Start creating dataset")

    create_dataset(
        video_dir_cave="/home/aicrowd/data/segments/FindCaveBrightness/stage_2",
        video_dir_expl="/home/aicrowd/data/segments/FindCaveBrightness/stage_1",
        stack_dir="/home/aicrowd/data/segments/FindCaveBrightness/rgb_images",
    )

    Logging.info("Finished creating dataset")
    Logging.info("Start training")

    train(
        stack_dir="/home/aicrowd/data/segments/FindCaveBrightness/rgb_images",
        model_dir="/home/aicrowd/train",
        model_name="impala", # either "impala" or "naturecnn"
        data_frac=0.1, # fraction of data to be used
        validation_frac=0.2, # fraction of loaded data to be used for validation
        report_rate=1,
    )

    Logging.info("Finished training")
