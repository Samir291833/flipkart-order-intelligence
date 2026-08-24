import os
import random
import numpy as np
import pandas as pd
import torch
import torchvision

from PIL import Image

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, TensorDataset

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support
)


# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_STATE = 42

DATA_DIR = "data"
CACHE_DIR = "data/cache"
MODEL_DIR = "models"
RESULTS_DIR = "results"
SAMPLE_DIR = "data/sample_images"

BATCH_SIZE = 64

NUM_CLASSES = 10
IMAGE_SIZE = 224

EPOCHS = 10
LEARNING_RATE = 0.001


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_STATE)


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)


# ============================================================
# FASHION-MNIST CLASSES
# ============================================================

class_names = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot"
]


# ============================================================
# CREATE OUTPUT DIRECTORIES
# ============================================================

os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(SAMPLE_DIR, exist_ok=True)


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

transform = transforms.Compose([

    # Fashion-MNIST is grayscale.
    # ResNet-18 expects 3 channels.
    transforms.Grayscale(num_output_channels=3),

    # ResNet-18 input size.
    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    # Convert PIL image to tensor.
    transforms.ToTensor(),

    # ImageNet normalization.
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# LOAD FASHION-MNIST
# ============================================================

full_train_dataset = datasets.FashionMNIST(
    root=DATA_DIR,
    train=True,
    download=True,
    transform=transform
)

test_dataset = datasets.FashionMNIST(
    root=DATA_DIR,
    train=False,
    download=True,
    transform=transform
)


print("\nOriginal Dataset")
print("----------------")
print(
    "Training images:",
    len(full_train_dataset)
)

print(
    "Test images:",
    len(test_dataset)
)


# ============================================================
# STRATIFIED TRAIN / VALIDATION SPLIT
# ============================================================

targets = full_train_dataset.targets.numpy()

indices = np.arange(
    len(full_train_dataset)
)

train_indices, val_indices = train_test_split(
    indices,
    test_size=6000,
    stratify=targets,
    random_state=RANDOM_STATE
)


train_dataset = Subset(
    full_train_dataset,
    train_indices
)

val_dataset = Subset(
    full_train_dataset,
    val_indices
)


print("\nActual Split")
print("------------")
print(
    "Training images:",
    len(train_dataset)
)

print(
    "Validation images:",
    len(val_dataset)
)

print(
    "Test images:",
    len(test_dataset)
)


# ============================================================
# VERIFY STRATIFICATION
# ============================================================

train_labels = targets[train_indices]
val_labels = targets[val_indices]
test_labels_array = test_dataset.targets.numpy()


print("\nClass Distribution")
print("------------------")

for class_id, class_name in enumerate(class_names):

    train_count = int(
        (train_labels == class_id).sum()
    )

    val_count = int(
        (val_labels == class_id).sum()
    )

    test_count = int(
        (test_labels_array == class_id).sum()
    )

    print(
        f"{class_id}: {class_name:<15} "
        f"train={train_count:<5} "
        f"val={val_count:<5} "
        f"test={test_count:<5}"
    )


# ============================================================
# DATA LOADERS
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# ============================================================
# LOAD PRETRAINED RESNET-18
# ============================================================

print("\nLoading pretrained ResNet-18...")

weights = torchvision.models.ResNet18_Weights.DEFAULT

model = torchvision.models.resnet18(
    weights=weights
)


# ============================================================
# FREEZE BACKBONE
# ============================================================

for parameter in model.parameters():
    parameter.requires_grad = False


# ============================================================
# FEATURE EXTRACTOR
# ============================================================

feature_extractor = torch.nn.Sequential(
    *list(model.children())[:-1]
)

feature_extractor = feature_extractor.to(device)

feature_extractor.eval()


# ============================================================
# FEATURE EXTRACTION FUNCTION
# ============================================================

def extract_features(loader, name):

    print(
        f"\nExtracting {name} features..."
    )

    all_features = []
    all_labels = []

    with torch.no_grad():

        for batch_index, (images, labels) in enumerate(
            loader
        ):

            images = images.to(device)

            features = feature_extractor(
                images
            )

            # ResNet-18 output:
            # [batch_size, 512, 1, 1]
            #
            # Convert to:
            # [batch_size, 512]

            features = features.flatten(1)

            all_features.append(
                features.cpu()
            )

            all_labels.append(
                labels
            )

            if (batch_index + 1) % 100 == 0:

                print(
                    f"Processed batches: "
                    f"{batch_index + 1}"
                )

    features = torch.cat(
        all_features,
        dim=0
    )

    labels = torch.cat(
        all_labels,
        dim=0
    )

    print(
        f"{name} feature shape:",
        features.shape
    )

    return features, labels


# ============================================================
# EXTRACT TRAINING FEATURES
# ============================================================

train_cache_path = (
    f"{CACHE_DIR}/train_features.pt"
)

val_cache_path = (
    f"{CACHE_DIR}/val_features.pt"
)


# Use existing cache if available.
# This prevents unnecessarily repeating the expensive
# ResNet-18 feature extraction step.

if (
    os.path.exists(train_cache_path)
    and os.path.exists(val_cache_path)
):

    print("\nCached features found.")
    print("Loading cached training features...")
    print("Loading cached validation features...")

    train_cache = torch.load(
        train_cache_path,
        map_location="cpu"
    )

    val_cache = torch.load(
        val_cache_path,
        map_location="cpu"
    )

    train_features = train_cache["features"]
    train_feature_labels = train_cache["labels"]

    val_features = val_cache["features"]
    val_feature_labels = val_cache["labels"]

    print(
        "Training feature shape:",
        train_features.shape
    )

    print(
        "Validation feature shape:",
        val_features.shape
    )

else:

    train_features, train_feature_labels = (
        extract_features(
            train_loader,
            "training"
        )
    )

    val_features, val_feature_labels = (
        extract_features(
            val_loader,
            "validation"
        )
    )

    torch.save(
        {
            "features": train_features,
            "labels": train_feature_labels
        },
        train_cache_path
    )

    torch.save(
        {
            "features": val_features,
            "labels": val_feature_labels
        },
        val_cache_path
    )


# ============================================================
# CACHE INFORMATION
# ============================================================

print("\nCached Features")
print("----------------")
print(
    "Training:",
    train_cache_path
)

print(
    "Validation:",
    val_cache_path
)


# ============================================================
# BUILD FEATURE DATASETS
# ============================================================

train_feature_dataset = TensorDataset(
    train_features,
    train_feature_labels
)

val_feature_dataset = TensorDataset(
    val_features,
    val_feature_labels
)


train_feature_loader = DataLoader(
    train_feature_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_feature_loader = DataLoader(
    val_feature_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# ============================================================
# CLASSIFIER HEAD
# ============================================================

classifier = torch.nn.Linear(
    512,
    NUM_CLASSES
)

classifier = classifier.to(device)


# ============================================================
# LOSS AND OPTIMIZER
# ============================================================

criterion = torch.nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    classifier.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# FEATURE EXTRACTION TRAINING
# ============================================================

print("\nFeature Extraction Training")
print("----------------------------")
print(
    "Optimizer: Adam"
)

print(
    "Learning rate:",
    LEARNING_RATE
)

print(
    "Batch size:",
    BATCH_SIZE
)

print(
    "Epochs:",
    EPOCHS
)


best_val_accuracy = 0.0
best_classifier_state = None


for epoch in range(EPOCHS):

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    classifier.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for features, labels in train_feature_loader:

        features = features.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = classifier(
            features
        )

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        optimizer.step()

        running_loss += (
            loss.item()
            * labels.size(0)
        )

        predictions = outputs.argmax(
            dim=1
        )

        correct += (
            predictions == labels
        ).sum().item()

        total += labels.size(0)

    train_loss = (
        running_loss / total
    )

    train_accuracy = (
        correct / total
    )


    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    classifier.eval()

    val_correct = 0
    val_total = 0
    val_loss_total = 0.0

    with torch.no_grad():

        for features, labels in val_feature_loader:

            features = features.to(device)
            labels = labels.to(device)

            outputs = classifier(
                features
            )

            loss = criterion(
                outputs,
                labels
            )

            val_loss_total += (
                loss.item()
                * labels.size(0)
            )

            predictions = outputs.argmax(
                dim=1
            )

            val_correct += (
                predictions == labels
            ).sum().item()

            val_total += labels.size(0)

    val_loss = (
        val_loss_total / val_total
    )

    val_accuracy = (
        val_correct / val_total
    )


    # --------------------------------------------------------
    # SAVE BEST VALIDATION HEAD
    # --------------------------------------------------------

    if val_accuracy > best_val_accuracy:

        best_val_accuracy = val_accuracy

        best_classifier_state = {
            key: value.detach().cpu().clone()
            for key, value
            in classifier.state_dict().items()
        }


    print(
        f"Epoch {epoch + 1:02d}/{EPOCHS} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Train Acc: {train_accuracy:.4f} | "
        f"Val Loss: {val_loss:.4f} | "
        f"Val Acc: {val_accuracy:.4f}"
    )


# ============================================================
# RESTORE BEST CLASSIFIER
# ============================================================

classifier.load_state_dict(
    best_classifier_state
)

classifier = classifier.to(device)

classifier.eval()


# ============================================================
# FEATURE EXTRACTION RESULT
# ============================================================

print("\nFeature Extraction Result")
print("-------------------------")

print(
    "Best validation accuracy:",
    round(best_val_accuracy, 4)
)

if best_val_accuracy >= 0.80:

    print(
        "Feature extraction alone achieved "
        "the required 80% validation accuracy."
    )

    print(
        "Fine-tuning is not required."
    )

else:

    print(
        "Validation accuracy is below 80%."
    )

    print(
        "Fine-tuning would be required."
    )


# ============================================================
# FINAL TEST EVALUATION
# ============================================================

print("\n" + "=" * 60)
print("FINAL TEST EVALUATION")
print("=" * 60)


# IMPORTANT:
# The test set has not been used for model selection.
# We extract its features only now for final evaluation.

test_features, test_feature_labels = extract_features(
    test_loader,
    "test"
)


# ------------------------------------------------------------
# TEST PREDICTIONS
# ------------------------------------------------------------

test_predictions = []

with torch.no_grad():

    for start in range(
        0,
        len(test_features),
        BATCH_SIZE
    ):

        batch_features = test_features[
            start:start + BATCH_SIZE
        ].to(device)

        outputs = classifier(
            batch_features
        )

        predictions = outputs.argmax(
            dim=1
        )

        test_predictions.append(
            predictions.cpu()
        )


test_predictions = torch.cat(
    test_predictions
)


# ------------------------------------------------------------
# NUMPY ARRAYS
# ------------------------------------------------------------

y_true = test_feature_labels.numpy()
y_pred = test_predictions.numpy()


# ============================================================
# TEST ACCURACY
# ============================================================

test_accuracy = accuracy_score(
    y_true,
    y_pred
)


print("\nFinal Test Accuracy")
print("-------------------")

print(
    "Accuracy:",
    round(test_accuracy, 4)
)

print(
    "Accuracy (%):",
    round(
        test_accuracy * 100,
        2
    ),
    "%"
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_true,
    y_pred,
    labels=list(range(NUM_CLASSES))
)


print("\nConfusion Matrix")
print("----------------")

print(
    "Rows = True class"
)

print(
    "Columns = Predicted class\n"
)


print(
    "      " +
    " ".join(
        f"{i:>6}"
        for i in range(NUM_CLASSES)
    )
)


for class_id, row in enumerate(cm):

    print(
        f"{class_id:>3} | " +
        " ".join(
            f"{value:>6}"
            for value in row
        )
    )


# ============================================================
# PER-CLASS METRICS
# ============================================================

precision, recall, f1, support = (
    precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=list(range(NUM_CLASSES)),
        zero_division=0
    )
)


print("\nPer-Class Metrics")
print("-----------------")

print(
    f"{'ID':<4}"
    f"{'Class':<16}"
    f"{'Precision':<12}"
    f"{'Recall':<12}"
    f"{'F1':<12}"
    f"{'Support':<8}"
)


for class_id, class_name in enumerate(
    class_names
):

    print(
        f"{class_id:<4}"
        f"{class_name:<16}"
        f"{precision[class_id]:<12.4f}"
        f"{recall[class_id]:<12.4f}"
        f"{f1[class_id]:<12.4f}"
        f"{support[class_id]:<8}"
    )


# ============================================================
# IDENTIFY CONFUSION PAIRS
# ============================================================

confusion_pairs = []


for true_class in range(NUM_CLASSES):

    for predicted_class in range(NUM_CLASSES):

        if true_class == predicted_class:
            continue

        count = cm[
            true_class,
            predicted_class
        ]

        confusion_pairs.append({
            "true_class": true_class,
            "predicted_class": predicted_class,
            "count": int(count)
        })


confusion_pairs.sort(
    key=lambda x: x["count"],
    reverse=True
)


print("\nTop 10 Confusion Pairs")
print("----------------------")


for rank, pair in enumerate(
    confusion_pairs[:10],
    start=1
):

    true_class = pair["true_class"]
    predicted_class = pair["predicted_class"]

    print(
        f"{rank}. "
        f"True = "
        f"{class_names[true_class]} "
        f"({true_class}) -> "
        f"Predicted = "
        f"{class_names[predicted_class]} "
        f"({predicted_class}) | "
        f"Count = {pair['count']}"
    )


# ============================================================
# TWO REQUIRED CONFUSION PAIRS
# ============================================================

top_pair_1 = confusion_pairs[0]
top_pair_2 = confusion_pairs[1]


print("\nTwo Most Common Confusion Pairs")
print("--------------------------------")


print(
    "Pair 1:",
    class_names[
        top_pair_1["true_class"]
    ],
    "->",
    class_names[
        top_pair_1["predicted_class"]
    ],
    "| Count:",
    top_pair_1["count"]
)


print(
    "Pair 2:",
    class_names[
        top_pair_2["true_class"]
    ],
    "->",
    class_names[
        top_pair_2["predicted_class"]
    ],
    "| Count:",
    top_pair_2["count"]
)


# ============================================================
# SAVE CONFUSION MATRIX
# ============================================================

confusion_matrix_path = (
    f"{RESULTS_DIR}/confusion_matrix.csv"
)


cm_dataframe = pd.DataFrame(
    cm,
    index=class_names,
    columns=class_names
)


cm_dataframe.to_csv(
    confusion_matrix_path
)


# ============================================================
# SAVE PER-CLASS METRICS
# ============================================================

metrics_path = (
    f"{RESULTS_DIR}/per_class_metrics.csv"
)


metrics_dataframe = pd.DataFrame({
    "class_id": range(NUM_CLASSES),
    "class_name": class_names,
    "precision": precision,
    "recall": recall,
    "f1": f1,
    "support": support
})


metrics_dataframe.to_csv(
    metrics_path,
    index=False
)


# ============================================================
# SAVE TEST ACCURACY
# ============================================================

accuracy_path = (
    f"{RESULTS_DIR}/test_accuracy.txt"
)


with open(
    accuracy_path,
    "w"
) as file:

    file.write(
        f"Test Accuracy: "
        f"{test_accuracy:.6f}\n"
    )

    file.write(
        f"Test Accuracy (%): "
        f"{test_accuracy * 100:.2f}%\n"
    )


# ============================================================
# SAVE COMPLETE MODEL
# ============================================================

# Put the trained classifier back onto the original
# ResNet-18 architecture.

model.fc = classifier

model = model.to(device)

model.eval()


model_path = (
    f"{MODEL_DIR}/product_classifier.pt"
)


torch.save(
    model.state_dict(),
    model_path
)


print("\nSaved Model")
print("-----------")
print(
    "Model:",
    model_path
)


# ============================================================
# VERIFY SAVED MODEL
# ============================================================

print("\nSaved Model Verification")
print("------------------------")


verification_model = torchvision.models.resnet18(
    weights=None
)

verification_model.fc = torch.nn.Linear(
    512,
    NUM_CLASSES
)

verification_model.load_state_dict(
    torch.load(
        model_path,
        map_location=device
    )
)

verification_model = (
    verification_model.to(device)
)

verification_model.eval()


# Verify predictions from the saved model
# using the already extracted test features.

verification_predictions = []


with torch.no_grad():

    for start in range(
        0,
        len(test_features),
        BATCH_SIZE
    ):

        batch_features = test_features[
            start:start + BATCH_SIZE
        ].to(device)

        outputs = verification_model.fc(
            batch_features
        )

        predictions = outputs.argmax(
            dim=1
        )

        verification_predictions.append(
            predictions.cpu()
        )


verification_predictions = torch.cat(
    verification_predictions
).numpy()


verification_accuracy = accuracy_score(
    y_true,
    verification_predictions
)


print(
    "Loaded model type:",
    type(verification_model).__name__
)

print(
    "Classifier type:",
    type(
        verification_model.fc
    ).__name__
)

print(
    "Original test accuracy:",
    round(
        test_accuracy,
        6
    )
)

print(
    "Reloaded model accuracy:",
    round(
        verification_accuracy,
        6
    )
)

print(
    "Accuracy difference:",
    round(
        abs(
            test_accuracy
            - verification_accuracy
        ),
        10
    )
)


# ============================================================
# EXPORT REAL TEST IMAGES
# ============================================================

print("\nExporting Sample Test Images")
print("----------------------------")


# The raw pixel data is available directly through
# Fashion-MNIST's .data tensor.
#
# These are REAL images from the official test split.

raw_test_data = test_dataset.data.numpy()
raw_test_targets = test_dataset.targets.numpy()


# Select at least one real image from several classes.
# Here we select the first occurrence of each of
# five different classes.

selected_indices = []

selected_classes = [
    0, 1, 2, 7, 9
]


for class_id in selected_classes:

    matches = np.where(
        raw_test_targets == class_id
    )[0]

    selected_indices.append(
        int(matches[0])
    )


for image_number, dataset_index in enumerate(
    selected_indices,
    start=1
):

    image_array = raw_test_data[
        dataset_index
    ]

    true_label = int(
        raw_test_targets[
            dataset_index
        ]
    )

    class_name = class_names[
        true_label
    ]

    # Make filename filesystem-safe.
    safe_class_name = (
        class_name
        .replace("/", "_")
        .replace(" ", "_")
    )

    filename = (
        f"{image_number:02d}_"
        f"{safe_class_name}.png"
    )

    image_path = os.path.join(
        SAMPLE_DIR,
        filename
    )

    image = Image.fromarray(image_array)

    image.save(
        image_path
    )

    print(
        f"Saved: {image_path} "
        f"| True label: {class_name}"
    )


# ============================================================
# SINGLE-IMAGE PREDICTION FUNCTION
# ============================================================

def load_product_classifier(
    model_path="models/product_classifier.pt"
):

    loaded_model = (
        torchvision.models.resnet18(
            weights=None
        )
    )

    loaded_model.fc = torch.nn.Linear(
        512,
        NUM_CLASSES
    )

    loaded_model.load_state_dict(
        torch.load(
            model_path,
            map_location=device
        )
    )

    loaded_model = (
        loaded_model.to(device)
    )

    loaded_model.eval()

    return loaded_model


def classify_product_image(
    image_path,
    model_path="models/product_classifier.pt"
):

    loaded_model = load_product_classifier(
        model_path
    )

    image = Image.open(
        image_path
    ).convert("L")

    image_tensor = transform(
        image
    )

    image_tensor = image_tensor.unsqueeze(
        0
    ).to(device)


    with torch.no_grad():

        logits = loaded_model(
            image_tensor
        )

        probabilities = torch.softmax(
            logits,
            dim=1
        )

        predicted_class = torch.argmax(
            probabilities,
            dim=1
        ).item()

        confidence = probabilities[
            0,
            predicted_class
        ].item()


    return {
        "category": class_names[
            predicted_class
        ],
        "confidence": confidence
    }


# ============================================================
# VERIFY SINGLE-IMAGE PREDICTION
# ============================================================

print("\nSingle-Image Prediction Verification")
print("------------------------------------")


sample_files = sorted(
    os.listdir(SAMPLE_DIR)
)


if len(sample_files) > 0:

    first_sample = os.path.join(
        SAMPLE_DIR,
        sample_files[0]
    )

    prediction_result = (
        classify_product_image(
            first_sample
        )
    )

    print(
        "Image:",
        first_sample
    )

    print(
        "Prediction:",
        prediction_result
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\nFinal Summary")
print("------------")

print("\nDataset")
print("Original training split:", "60,000")
print("Actual training split:", "54,000")
print("Validation split:", "6,000")
print("Test split:", "10,000")

print("\nPerformance")
print(
    "Best validation accuracy:",
    f"{best_val_accuracy * 100:.2f}%"
)

print(
    "Final test accuracy:",
    f"{test_accuracy * 100:.2f}%"
)

print("\nSaved Files")
print("Model:", model_path)
print("Confusion matrix:", confusion_matrix_path)
print("Per-class metrics:", metrics_path)
print("Sample images:", SAMPLE_DIR)

print("\nFine-tuning required:",
    "NO" if best_val_accuracy >= 0.80 else "YES")