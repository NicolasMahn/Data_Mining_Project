from assignment4 import assignment4
from test_clustering import test_clustering
from test_network_analysis import test_community_detection
from test_projection import test_projection


def main():
    data_mining_project()

    data_mining_assignment4()


def data_mining_project():
    test_clustering()

    test_projection()

    test_community_detection()


def data_mining_assignment4():

    assignment4()


if __name__ == "__main__":
    main()

