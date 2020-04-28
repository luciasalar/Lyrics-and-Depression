"""this script is to merge all the data."""
import glob
import os
import pandas as pd


class MergeData:
    """merge all the csv file in the directory."""

    def __init__(self):
        """Define varibles."""
        self.path = '/afs/inf.ed.ac.uk/user/s16/s1690903/share/QuoteAndDepression/data/'
        self.path1 = '/disk/data/share/s1690903/predicting_depression_symptoms/data/'
        self.dir = self.path + "all_search/"
        self.save_dir = self.path + "all_search.csv"

    def get_file_names(self):
        """Get all the file names and merge them."""
        df = pd.DataFrame()
        os.chdir(self.dir)
        for files in glob.glob("*.csv"):
            df = pd.concat([df, pd.read_csv(files).iloc[:, 0:]], axis=0)
        return df

    def save_file(self):
        """Save the merged document as csv."""
        all_doc = self.get_file_names()
        all_doc.to_csv(self.save_dir)


m = MergeData()
f = m.save_file()
