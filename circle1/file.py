from pickle import dump, load
import os

def save_training_trainable_output(output):
    """
    Save the training of the trainable QML model output to file.
    """
    if not os.path.exists("./data"):
        os.makedirs("./data")
    dump(output, open("./data/train_trainable.p", "wb"))
    return

def load_training_trainable_output():
    """
    Load the training of the trainable QML model output from file.
    """
    output = load(open("./data/train_trainable.p", "rb"))
    return output

def save_training_non_trainable_output(output):
    """
    Save the training of the non trainable QML model output to file.
    """
    if not os.path.exists("./data"):
        os.makedirs("./data")
    dump(output, open("./data/train_non_trainable.p", "wb"))
    return

def load_training_non_trainable_output():
    """
    Load the training of the non trainable QML model output from file.
    """
    output = load(open("./data/train_non_trainable.p", "rb"))
    return output

def save_trainable_robustness_output(output):
    """
    Save the robustness output for the trainable encoding QML model to file.
    """
    if not os.path.exists("./data"):
        os.makedirs("./data")
    dump(output, open("./data/robustness_trainable.p", "wb"))
    return

def load_trainable_robustness_output():
    """
    Load the robustness output for the trainable encoding QML model from file.
    """
    output = load(open("./data/robustness_trainable.p", "rb"))
    return output

def save_non_trainable_robustness_output(output):
    """
    Save the robustness output for the non trainable encoding QML model to file.
    """
    if not os.path.exists("./data"):
        os.makedirs("./data")
    dump(output, open("./data/robustness_non_trainable.p", "wb"))
    return

def load_non_trainable_robustness_output():
    """
    Load the robustness output for the non trainable encoding QML model from file.
    """
    output = load(open("./data/robustness_non_trainable.p", "rb"))
    return output

def save_trainable_generalization_output(output):
    """
    Save the generalization output for the trainable encoding QML model to file.
    """
    if not os.path.exists("./data"):
        os.makedirs("./data")
    dump(output, open("./data/generalization_trainable.p", "wb"))
    return

def load_trainable_generalization_output():
    """
    Load the generalization output for the trainable encoding QML model from file.
    """
    output = load(open("./data/generalization_trainable.p", "rb"))
    return output

def save_non_trainable_generalization_output(output):
    """
    Save the generalization output for the non trainable encoding QML model to file.
    """
    if not os.path.exists("./data"):
        os.makedirs("./data")
    dump(output, open("./data/generalization_non_trainable.p", "wb"))
    return

def load_non_trainable_generalization_output():
    """
    Load the generalization output for the non trainable encoding QML model from file.
    """
    output = load(open("./data/generalization_non_trainable.p", "rb"))
    return output