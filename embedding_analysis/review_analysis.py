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

class ReviewAnalysis(EmbeddingsAnalysis):
    def __init__(self, update :bool=False, debug :bool=False, batch_size :int=100):

        super().__init__(["reviews"], False, update, debug, batch_size)

    def get_positive_negative_labels(self, refresh_labels=False):
        labels = []
        if not refresh_labels:
            labels = self.load_precalculated_data("positive_negative_labels")
        if len(labels) == 0 or len(labels) != len(self.topic_labels):
            labels = self.get_positive_negative_labels_from_id()
            self.save_precalculated_data(labels, "positive_negative_labels")
        return labels

    def get_hotel_name_labels(self, refresh_labels=False):
        return self.get_csv_labels("Hotel_Name", refresh_labels)

    def get_average_score_labels(self, refresh_labels=False):
        return self.get_csv_labels("Average_Score", refresh_labels)

    def get_reviewer_score_labels(self, refresh_labels=False):
        return self.get_csv_labels("Reviewer_Score", refresh_labels)

    def get_reviewer_nationality_labels(self, refresh_labels=False):
        return self.get_csv_labels("Reviewer_Nationality", refresh_labels)

    def get_total_number_of_reviews_reviewer_has_given_labels(self, refresh_labels=False):
        return self.get_csv_labels("Total_Number_of_Reviews_Reviewer_Has_Given", refresh_labels)

    def get_tags_labels(self, refresh_labels=False):
        labels = []
        if not refresh_labels:
            labels = self.load_precalculated_data(f"tags_labels")
        if len(labels) == 0 or len(labels) != len(self.topic_labels):
            labels = self.load_labels_from_csv("Tags")
            labels = [ast.literal_eval(label) if label is not None else None for label in labels]
            self.save_precalculated_data(labels, f"tags_labels")
        return labels

    def get_hotel_country_labels(self, refresh_labels=False):
        labels = []
        if not refresh_labels:
            labels = self.load_precalculated_data(f"hotel_country_labels")
        if len(labels) == 0 or len(labels) != len(self.topic_labels):
            labels = self.extract_hotel_countries_from_csv()
            self.save_precalculated_data(labels, f"hotel_country_labels")
        return labels

    def get_csv_labels(self, column_name, refresh_labels=False):
        labels = []
        if not refresh_labels:
            labels = self.load_precalculated_data(f"{column_name.lower()}_labels")
        if len(labels) == 0 or len(labels) != len(self.topic_labels):
            labels = self.load_labels_from_csv(column_name)
            self.save_precalculated_data(labels, f"{column_name.lower()}_labels")
        return labels

    def get_trip_type_labels(self, refresh_labels=False):
        tags = self.get_tags_labels(refresh_labels)
        if tags is None:
            return None
        labels = []
        for tag in tags:
            if tag is None:
                labels.append(None)
            elif " Leisure trip " in tag:
                labels.append("Leisure")
            elif " Business trip " in tag:
                labels.append("Business")
            else:
                labels.append("Unknown")
        return labels

    def get_booked_mobile_labels(self, refresh_labels=False):
        tags = self.get_tags_labels(refresh_labels)
        if tags is None:
            return None
        labels = []
        for tag in tags:
            if tag is None:
                labels.append(None)
            elif " Submitted from a mobile device " in tag:
                labels.append("Mobile")
            else:
                labels.append("Unknown")
        return labels

    def get_traveler_type_labels(self, refresh_labels=False):
        tags = self.get_tags_labels(refresh_labels)
        if tags is None:
            return None
        labels = []
        for tag in tags:
            if tag is None:
                labels.append(None)
            elif " Solo traveler " in tag:
                labels.append("Solo")
            elif " Couple " in tag:
                labels.append("Couple")
            elif " Group " in tag:
                labels.append("Group")
            elif " Family with young children " in tag:
                labels.append("Family with young children")
            elif " Family with older children " in tag:
                labels.append("Family with older children")
            else:
                labels.append("Unknown")
        return labels

    def get_stay_duration_labels(self, refresh_labels=False):
        tags = self.get_tags_labels(refresh_labels)
        if tags is None:
            return None
        labels = []
        for tag in tags:
            if tag is None:
                labels.append(None)
            else:
                for t in tag:
                    match = re.search(r" Stayed (\d+) nights? ", t)
                    if match:
                        labels.append(int(match.group(1)))
                        break
                else:
                    labels.append("Unknown")
        return labels

    def get_review_text_with_br(self, n=40, refresh_text=False):
        text = []
        if not refresh_text:
            text = self.load_precalculated_data(f"review_text")
        if len(text) == 0 or len(text) != len(self.embeddings):
            review_type = self.get_positive_negative_labels(refresh_labels=refresh_text)
            if review_type is None:
                return None
            negative_reviews = self.load_labels_from_csv("Negative_Review")
            positive_reviews = self.load_labels_from_csv("Positive_Review")
            for i in range(len(review_type)):
                if review_type[i] == "negative":
                    text.append(self.insert_br_every_n_chars(negative_reviews[i], n))
                elif review_type[i] == "positive":
                    text.append(self.insert_br_every_n_chars(positive_reviews[i], n))
                else:
                    text.append(None)

            self.save_precalculated_data(text, f"review_text")
        return text

    def load_labels_from_csv(self, column_name):
        labels = []
        data = self.get_hotel_csv()
        for i in range(len(self.topic_labels)):
            if "Reviews" == self.topic_labels[i]:
                csv_index = self.extract_number(self.id_labels[i])
                labels.append(data[column_name].values[csv_index])
            else:
                labels.append(None)
        del data
        return labels

    def insert_br_every_n_chars(self, text, n):
        words = text.split()
        char_count = 0
        result = []
        for word in words:
            if char_count + len(word) > n:
                result.append('<br>')
                char_count = 0
            result.append(word)
            char_count += len(word) + 1  # +1 for the space
        return ' '.join(result)

    def get_positive_negative_labels_from_id(self):
        labels = []
        for i in range(len(self.topic_labels)):
            if "Reviews" == self.topic_labels[i]:
                if "negative" in self.id_labels[i].lower():
                    labels.append("negative")
                elif "positive" in self.id_labels[i].lower():
                    labels.append("positive")
                else:
                    labels.append(None)
            else:
                labels.append(None)
        return labels

    def get_hotel_csv(self):
        base_path = os.path.abspath(os.path.dirname(__file__))  # Get current file directory
        file_path = os.path.join(base_path, "..", "data", "reviews", "Hotel_Reviews.csv")
        data = pd.read_csv(file_path)
        return data

    def extract_hotel_countries_from_csv(self):
        data = self.get_hotel_csv()
        addresses = data["Hotel_Address"].values
        possible_countries = ["France", "Netherlands", "Spain", "Italy", "Austria", "United Kingdom"]
        labels = []
        for i in range(len(self.topic_labels)):
            if "Reviews" == self.topic_labels[i]:
                csv_index = self.extract_number(self.id_labels[i])
                address = addresses[csv_index]
                for country in possible_countries:
                    if country in address:
                        labels.append(country)
                        break
                    if country == possible_countries[-1]:
                        labels.append("Unknown Country")
            else:
                labels.append(None)
        del data
        return labels

    def extract_number(self, id_string):
        match = re.search(r'\|(\d+)_', id_string)
        if match:
            return int(match.group(1))
        return None