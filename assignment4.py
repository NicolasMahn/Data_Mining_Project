import math
from tqdm import tqdm
import pandas as pd
import plotly.express as px
import plotly.subplots as sp
import numpy as np

from algorithms.clustering import get_available_clustering_algorithms
from algorithms.distance_measures import distance_pipeline
from algorithms.projection import get_available_projection_algorithms
from embedding_analysis.language_analysis import LanguageAnalysis
from embedding_analysis.review_analysis import ReviewAnalysis
from embedding_analysis.speech_analysis import SpeechAnalysis

WHITE = "\033[97m"
BLUE = "\033[34m"
GREEN = "\033[32m"
ORANGE = "\033[38;5;208m"
PINK = "\033[38;5;205m"
RED = "\033[31m"
RESET = "\033[0m"

NO_SAMMON_MAPPING = True  # Sammon mapping is disabled by default as it simply takes to long to compute
NO_HOME_BREW_DBSCAN = True

def assignment4():

    analyse_reviews()

    analyse_languages()

    analyse_speech()


def analyse_reviews():
    ra = ReviewAnalysis()

    comparing_review(ra)


def analyse_languages():
    la = LanguageAnalysis()

    comparing_languages(la)

def analyse_speech():

    sa = SpeechAnalysis()

    comparing_speech(sa)


def comparing_review(ra):
    color = 'Review Type'

    review_type = ra.get_positive_negative_labels()
    hotel_name = ra.get_hotel_name_labels()
    hotel_country = ra.get_hotel_country_labels()
    average_score = ra.get_average_score_labels()
    reviewer_score = ra.get_reviewer_score_labels()
    reviewer_nationality = ra.get_reviewer_nationality_labels()
    total_number_of_reviews_reviewer_has_given = ra.get_total_number_of_reviews_reviewer_has_given_labels()
    trip_type = ra.get_trip_type_labels()
    booked_mobile = ra.get_booked_mobile_labels()
    traveler_type = ra.get_traveler_type_labels()
    stay_duration = ra.get_stay_duration_labels()
    review_text = ra.get_review_text_with_br()

    df = pd.DataFrame({
        'x': None,
        'y': None,
        'Cluster': None,
        'Review Type': review_type,
        'Hotel Name': hotel_name,
        'Hotel Country': hotel_country,
        'Average Hotel Score': average_score,
        'Review Score': reviewer_score,
        'Reviewer Nationality': reviewer_nationality,
        'Total Number of Reviews Reviewer Has Given': total_number_of_reviews_reviewer_has_given,
        'Trip Type': trip_type,
        'Submitted Mobile': booked_mobile,
        'Traveler Type': traveler_type,
        'Stay Duration': stay_duration,
        'Review Text': review_text
    })
    plot_name = "Dynamic Grid of Reviews, displaying positive vs negative Reviews"
    projection_grid(ra, df, color, plot_name)
    print(f"{BLUE} Positive vs Negative Reviews")
    print(WHITE, end="")
    print("In comparing the different visualizations for the reviews, I'd have to say that T-SNE performed better \n"
          "then the rest. While all methods manage to show a clear difference between positive and negative reviews \n"
          "T-SNE is the only one that has meaningful subdivisions. This is not immediately obvious in this \n"
          "visualization as it can only really be seen if hovering over the points and reading the review text.")
    print(RESET)

    plot_name = "Dynamic Grid of Reviews, displaying the Class Preservation"
    projection_grid(ra, df, color, plot_name, class_preservation=True)
    print(f"{BLUE} Class Preservation on Reviews")
    print(WHITE, end="")
    print("This shows that all methods preformed quite well in separating the classes. Now a perfect \n"
          "separation can not be expected especially since some negative and positive reviews are very similar.")
    print(RESET)

    params = {
        'epsilon': 0.6,
        'min_numbs': 7,
        "n_clusters": 2,
    }

    plot_name = "Dynamic Grid of Reviews comparing different Clustering Algorithms"
    clustering_grid(ra, df, params, plot_name)
    print(f"{BLUE} Comparing Clustering on Reviews")
    print(WHITE, end="")
    print("Sadly the reviews do not lend themselves to being clustered. Aside from that it is hard to say how many \n"
          "clusters the reviews should have, although they are labeled. In my opinion no clustering technique \n"
          "performed well. Obviously DBSCAN performed the worst, i have a method of getting the optimal epsilon and \n"
          "min_samples values (with in a set of values), but apparently they didn't lead to any satisfactory results.")
    print(RESET)

    plot_name = "Dynamic Grid comparing Clustering Preservation on Reviews"
    clustering_grid(ra, df, params, plot_name, cluster_preservation=True)
    print(f"{BLUE} Comparing Clustering Preservation on Reviews")
    print(WHITE, end="")
    print("I don't know what to say theres seems to have been an arbitrary line that was drawn by the algorithms")
    print(RESET)


def comparing_languages(la):
    color = 'Language'
    language = la.get_language()
    word = la.get_word_labels()
    df = pd.DataFrame({
        'x': None,
        'y': None,
        'Cluster': None,
        'Language': language,
        'Word': word
    })
    plot_name = "Dynamic Grid of Languages, displaying English, German, Swedish and Portuguese"
    projection_grid(la, df, color, plot_name)

    print(f"{BLUE} English, German, Swedish and Portuguese")
    print(WHITE, end="")
    print("In T-SNE all languages are clearly separated, this has to go to T-SNE. PCA did a decent job of separating \n"
          "german from portuguese and english, but english swedish and portuguese are layerd on top of one another. \n"
          "MDS did not do a good job of separating the languages either although it did perform better then PCA.")
    print(RESET)

    plot_name = "Dynamic Grid of Reviews, displaying the Class Preservation"
    projection_grid(la, df, color, plot_name, class_preservation=True)
    print(f"{BLUE} Class Preservation on Reviews")
    print(WHITE, end="")
    print("Here T-SNE shines too it is the only one where one would guess 4 different languages are being displayed.")
    print(RESET)

    params = {
        'epsilon': 1.3,
        'min_numbs': 3,
        "n_clusters": 4,
    }

    plot_name = "Dynamic Grid of Languages comparing different Clustering Algorithms"
    clustering_grid(la, df, params, plot_name)
    print(f"{BLUE} Comparing Clustering on Reviews")
    print(WHITE, end="")
    print("I think th clustering for the languages is quite decent for K-Means and Hierarchical Clustering. \n"
          "Both show that they have somewhat reasonably seperated the languages. DBSCAN performs poorly again. \n"
          "There are probably better parameters, but they couldn't be found.")
    print(RESET)

    plot_name = "Dynamic Grid comparing Clustering Preservation on Languages"
    clustering_grid(la, df, params, plot_name, cluster_preservation=True)
    print(f"{BLUE} Comparing Clustering Preservation on Reviews")
    print(WHITE, end="")
    print("This shows that there is a pretty good cluster preservation with T-SNE and Hierarchical Clustering")
    print(RESET)


def comparing_speech(sa):
    color = 'Word Type'

    word_type = sa.get_word_type()
    word = sa.get_word_labels()
    df = pd.DataFrame({
        'x': None,
        'y': None,
        'Cluster': None,
        'Word Type': word_type,
        'Word': word
    })
    plot_name = "Dynamic Grid of Word Types, displaying Verbs, Nouns and Adjectives"
    projection_grid(sa, df, color, plot_name)

    print(f"{BLUE} Verbs, Nouns and Adjectives")
    print(WHITE, end="")
    print("The Verbs, Nouns and Adjectives are relatively clearly separated in PCA (and somewhat in MDS). In T-SNE \n"
          "there is an interesting picture showing where verbs and adjectives with a negative sentiment cluster at \n"
          "the bottom. Here I find it hard to select a favourite, but ir is clear the T-SNE is showing the most\n"
          "detail.\n")
    print(RESET)

    plot_name = "Dynamic Grid of Word Types, displaying the Class Preservation"
    projection_grid(sa, df, color, plot_name, class_preservation=True)

    print(f"{BLUE} Class Preservation on Word Types")
    print(WHITE, end="")
    print("This shows that all methods performed poorly at preserving the classes.\n")
    print(RESET)

    params = {
        'epsilon': 1.3,
        'min_numbs': 19,
        "n_clusters": 3,
    }

    plot_name = "Dynamic Grid of Word Types comparing different Clustering Algorithms"
    clustering_grid(sa, df, params, plot_name)
    print(f"{BLUE} Comparing Clustering on Reviews")
    print(WHITE)
    print("The clustering is again rubbish for DBSCAN. But K-Means and Hierarchical Clustering did a decent job of \n"
          "separating something. As the bottom cluster in T-SNE do a pear to be the words with a negative sentiment.")
    print(RESET)

    plot_name = "Dynamic Grid comparing Clustering Preservation on Word Types"
    clustering_grid(sa, df, params, plot_name, cluster_preservation=True)
    print(f"{BLUE} Comparing Clustering Preservation on Reviews")
    print(WHITE)
    print("The cluster preservation is way better for the words then for the reviews. especially with PCA and K-Means.")
    print(RESET)


def projection_grid(ea, df, color, plot_name, class_preservation=False, k=5):
    # Retrieve the available projection algorithms
    available_projection_algorithms = get_available_projection_algorithms()
    if NO_SAMMON_MAPPING and 'Sammon Mapping' in available_projection_algorithms:
        available_projection_algorithms.remove('Sammon Mapping')

    num_projection_algorithms = len(available_projection_algorithms)

    # Calculate grid dimensions
    grid_size = math.ceil(math.sqrt(num_projection_algorithms))

    # Initialize plotly subplots with calculated grid size
    fig = sp.make_subplots(rows=grid_size, cols=grid_size, subplot_titles=available_projection_algorithms)

    if color in get_available_clustering_algorithms():
        clusters_int = ea.get_clusters(color)
        clusters = []
        for d in clusters_int:
            if d == -1:
                clusters.append('Noise')
            else:
                clusters.append(str(d))

        df["Cluster"] = clusters

    if color == 'Review Type':
        color_discrete_map = {'positive': 'blue', 'negative': 'red'}
    else:
        color_discrete_map = None

    if color in get_available_clustering_algorithms():
        color = 'Cluster'

    hover_data = [col for col in df.columns if col not in ['x', 'y'] and df[col].notna().any()]

    # Iterate over each algorithm and create a subplot
    if class_preservation:
        bar_format = (f"{WHITE}⌛  Plotting Projection Methods with Class Preservation...   "
                      f"{{l_bar}}{PINK}{{bar}}{WHITE}{{r_bar}}{RESET}")
    else:
        bar_format = f"{WHITE}⌛  Plotting Projection Methods...   {{l_bar}}{ORANGE}{{bar}}{WHITE}{{r_bar}}{RESET}"
    with tqdm(total=num_projection_algorithms, bar_format=bar_format) as pbar:
        for idx, algorithm in enumerate(available_projection_algorithms):

            # Generate the reduced points for each algorithm
            points = ea.get_reduced_similarities(algorithm=algorithm)

            df['x'], df['y'] = points[:, 0], points[:, 1]

            if class_preservation:
                # Compute class preservation for each point
                class_preservation = []
                for i, point in df.iterrows():
                    distances = np.linalg.norm(points - point[['x', 'y']].values.astype(float), axis=1)
                    nearest_indices = np.argsort(distances)[1:k + 1]
                    same_class_count = sum(df.iloc[nearest_indices][color] == point[color])
                    class_preservation.append(same_class_count / k)
                df['Class Preservation'] = class_preservation

                scatter = px.scatter(df, x='x', y='y', color='Class Preservation', hover_data=hover_data,
                                     color_continuous_scale='Cividis')
            else:
                scatter = px.scatter(df, x='x', y='y', color=color, hover_data=hover_data,
                                     color_discrete_map=color_discrete_map)

            # Determine subplot position
            row = (idx // grid_size) + 1
            col = (idx % grid_size) + 1

            # Add the scatter plot to the subplot
            for trace in scatter.data:
                fig.add_trace(trace, row=row, col=col)
            pbar.update(1)

    # Update layout and axis visibility
    fig.update_layout(height=900, width=900, title_text=plot_name)
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)

    fig.show()


def clustering_grid(ea, df, params, plot_name, cluster_preservation=False, k=5):
    # Retrieve the available projection algorithms
    available_projection_algorithms = get_available_projection_algorithms()
    if NO_SAMMON_MAPPING and 'Sammon Mapping' in available_projection_algorithms:
        available_projection_algorithms.remove('Sammon Mapping')

    get_clustering_algorithms = get_available_clustering_algorithms()
    if NO_HOME_BREW_DBSCAN:
        get_clustering_algorithms.remove('DBSCAN')
    else:
        get_clustering_algorithms.remove('Sklearn DBSCAN')

    num_projection_algorithms = len(available_projection_algorithms)
    num_clustering_algorithms = len(get_clustering_algorithms)

    fig = sp.make_subplots(rows=num_projection_algorithms, cols=num_clustering_algorithms,
                           subplot_titles=[f"{proj} + {clust}" for proj in available_projection_algorithms for clust in
                                           get_clustering_algorithms])

    hover_data = [col for col in df.columns if col not in ['x', 'y'] and df[col].notna().any()]

    # Iterate over each algorithm and create a subplot
    bar_format = f"{WHITE}⌛  Plotting Clustering Methods...   {{l_bar}}{ORANGE}{{bar}}{WHITE}{{r_bar}}{RESET}"
    with tqdm(total=num_clustering_algorithms*num_projection_algorithms, bar_format=bar_format) as pbar:
        for row_idx, projection_algorithm in enumerate(available_projection_algorithms):
            for col_idx, clustering_algorithm in enumerate(get_clustering_algorithms):
                # Generate the clusters for each clustering algorithm
                clusters_int = ea.get_clusters(algorithm=clustering_algorithm, params=params)
                clusters = []
                for d in clusters_int:
                    if d == -1:
                        clusters.append('Noise')
                    else:
                        clusters.append(str(d))

                df['Clusters'] = clusters

                hover_data.append('Clusters')


                # Generate the reduced points for each projection algorithm
                points = ea.get_reduced_similarities(algorithm=projection_algorithm)
                df['x'], df['y'] = points[:, 0], points[:, 1]

                if cluster_preservation:
                    cluster_preservation = []
                    for i, point in df.iterrows():
                        distances = np.linalg.norm(points - point[['x', 'y']].values.astype(float), axis=1)
                        nearest_indices = np.argsort(distances)[1:k + 1]
                        same_cluster_count = sum(df.iloc[nearest_indices]['Clusters'] == point['Clusters'])
                        cluster_preservation.append(same_cluster_count / k)

                    df['Cluster Preservation'] = cluster_preservation
                    scatter = px.scatter(df, x='x', y='y', color='Cluster Preservation', hover_data=hover_data)
                else:
                    scatter = px.scatter(df, x='x', y='y', color='Clusters', hover_data=hover_data)


                # Add the scatter plot to the subplot
                for trace in scatter.data:
                    fig.add_trace(trace, row=row_idx + 1, col=col_idx + 1)
                pbar.update(1)

        # Update layout and axis visibility
    fig.update_layout(height=900, width=900, title_text=plot_name)
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)

    fig.show()


if __name__ == '__main__':
    assignment4()