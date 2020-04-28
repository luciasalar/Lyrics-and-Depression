"""This script is to identify which post does not have search result."""
import pandas as pd
from QuoteDetector import *


class MissingData:
    """Identify which post does not have search result."""

    def __init__(self):
        """Define varibles."""
        self.path = '/afs/inf.ed.ac.uk/user/s16/s1690903/share/QuoteAndDepression/'
        self.sp = SelectText()

    def read_file(self):
        """Read search results."""
        search_re = pd.read_csv('data/all_search.csv')
        search_re2 = pd.read_csv('data/all_search2.csv')
        # concantenate files
        results = search_re.append(search_re2, sort=False)
        return results

    def unique_re(self):
        """Unique search results."""
        results = self.read_file()
        # get unique rows
        unique_re = results.drop_duplicates(subset='textID', keep='first')
        return unique_re

    def missing(self):
        """Missing results."""
        participants = self.sp.selected_text()
        participants = participants[['textID', 'text']]
        # all search results
        unique_re = m.unique_re()
        unique_re = unique_re[['textID', 'text']]
        # merge search result with all participants posts return difference
        miss_re = pd.concat([unique_re, participants]).drop_duplicates(subset='textID', keep=False)
        miss_re.to_csv('data/missing_search.csv')
        return miss_re

m = MissingData()
p = m.missing()


