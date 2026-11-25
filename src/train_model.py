"""Command-line training entry point for the Melody2Chord Random Forest baseline."""
from __future__ import annotations

from collections import Counter

from sklearn.model_selection import train_test_split

from src.dataset import DatasetValidationError, load_training_dataset, validate_training_dataset
from src.evaluate_model import evaluate_model
from src.feature_extraction import FeatureExtractionError, dataframe_to_features
from src.model import RANDOM_STATE, save_model, train_model


def _split_data(features, labels):
    """Use stratification only when every class can appear in train and test sets."""
    counts = Counter(labels)
    class_count = len(counts)
    smallest_class = min(counts.values())
    if class_count < 2:
        raise DatasetValidationError("Training requires at least two chord classes.")
    if smallest_class < 3:
        raise DatasetValidationError(
            "Each chord class needs at least 3 real examples for a meaningful stratified train/test split."
        )
    test_size = max(0.20, class_count / len(labels))
    if test_size >= 0.5:
        raise DatasetValidationError("Dataset is too small for a meaningful holdout set; add more examples per chord class.")
    return train_test_split(features, labels, test_size=test_size, random_state=RANDOM_STATE, stratify=labels)


def main() -> None:
    """Load, validate, train, evaluate, and save the baseline model."""
    print("Loading dataset...")
    raw_dataset = load_training_dataset()
    print("Validating dataset...")
    dataset, report = validate_training_dataset(raw_dataset)
    print(f"Loaded records: {report['loaded']}")
    print(f"Removed invalid records: {report['invalid']}")
    print(f"Removed duplicate records: {report['duplicates']}")
    print(f"Usable records: {report['usable']}")
    print(f"Available chord classes: {', '.join(report['available_classes'])}")
    print(f"Class counts: {report['class_counts']}")

    print("Preparing features...")
    features = dataframe_to_features(dataset)
    labels = dataset["chord_label"]
    X_train, X_test, y_train, y_test = _split_data(features, labels)
    print(f"Training records: {len(X_train)}")
    print(f"Testing records: {len(X_test)}")

    print("Training model...")
    model = train_model(X_train, y_train)
    print("Evaluating model...")
    evaluate_model(model, X_test, y_test)
    model_path = save_model(model)
    print(f"Model saved successfully: {model_path}")


if __name__ == "__main__":
    try:
        main()
    except (DatasetValidationError, FeatureExtractionError) as error:
        print(f"Training stopped: {error}")
