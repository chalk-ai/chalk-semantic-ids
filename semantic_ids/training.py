"""Compatibility imports for the native functions in the training stage folders.

Import after the CLI sets the saved Dataset names and checkpoint namespace.
Importing creates native function definitions; it does not submit training.
"""
from .stages import load_stage

train_tokenizer = load_stage("train").train_tokenizer
train_recommender = load_stage("train-recommender").train_recommender
