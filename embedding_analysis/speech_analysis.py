import ast
import math
import pickle
import re
import shutil

import numpy as np
import os

import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
from tqdm import tqdm

from algorithms.clustering import clustering_pipeline
from algorithms.projection import projection_pipeline
from .embbedings_analysis import EmbeddingsAnalysis
from .embedding_function import get_embedding_function
from .populate_database import DatabaseManager
from util import load_config

PINK = "\033[38;5;205m"
RESET = "\033[0m"
ORANGE = "\033[38;5;208m"
WHITE = "\033[97m"
BLUE = "\033[34m"

class SpeechAnalysis(EmbeddingsAnalysis):
    def __init__(self, update :bool=False, debug :bool=False, batch_size :int=100):

        super().__init__(["speech"], True, update, debug, batch_size)

    def get_word_type(self, refresh_labels=False):
        labels = []
        if not refresh_labels:
            labels = self.load_precalculated_data("word_type_labels")
        if len(labels) == 0 or len(labels) != len(self.topic_labels):
            labels = self.extract_word_type_from_id()
            self.save_precalculated_data(labels, "word_type_labels")
        return labels

    def extract_word_type_from_id(self):
        ids = self.get_id_labels()
        word_types = []
        for i in ids:
            if "verb" in i:
                word_types.append("Verb")
            elif "noun" in i:
                word_types.append("Noun")
            elif "adjective" in i:
                word_types.append("Adjective")
        return word_types
