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

class LanguageAnalysis(EmbeddingsAnalysis):
    def __init__(self, update :bool=False, debug :bool=False, batch_size :int=100):

        super().__init__(["languages"], True, update, debug, batch_size)

    def get_language(self, refresh_labels=False):
        labels = []
        if not refresh_labels:
            labels = self.load_precalculated_data("language_labels")
        if len(labels) == 0 or len(labels) != len(self.get_id_labels()):
            labels = self.extract_language_from_id()
            self.save_precalculated_data(labels, "language_labels")
        return labels

    def extract_language_from_id(self):
        ids = self.get_id_labels()
        languages = []
        for i in ids:
            if "english" in i:
                languages.append("English")
            elif "german" in i:
                languages.append("German")
            elif "swedish" in i:
                languages.append("Swedish")
            elif "portuguese" in i:
                languages.append("Portuguese")
            else:
                languages.append("Other")
        return languages
