# This file is dedicated to classes constructing objects fed to the neural network
from numpy import dtype
from DataHandler import *
from datetime import datetime
import math


# Class used for collecting aggregated info of a single item for later purposes
class ItemData:
    def __init__(self, id_item, category, number_of_clicks_init, clicktime_init):
        self.id_item = id_item
        self.category = category
        self.number_of_clicks = number_of_clicks_init
        self.clicktime_total = clicktime_init
        self.number_of_buys = 0

    # Add clicks and time to item, so no direct manipulation outside class is needed
    def update_item(self, additional_clicks, additional_time):
        self.number_of_clicks += additional_clicks
        self.clicktime_total += additional_time


    # Initialized counts of item buys for each item
    # Takes data from session dictionary, session.bought_items list
    @staticmethod
    def initialize_item_buys(item_dict, session_dict):

        for session in session_dict.keys():

            bought = session_dict[session].bought_items

            for item in bought:
                item_dict[item].number_of_buys += 1