import logging
import pandas as pd

from abc import ABC, abstractmethod
from sklearn import preprocessing

import util


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


# Experiment classes
class Experiment(ABC):
    def __init__(self, verbose=False):
        self._verbose = verbose
        self._logger = logging.getLogger(self.__class__.__name__)

    @abstractmethod
    def run(self):
        """
        Run the experiment
        """
        pass

    def log(self, msg, *args):
        if self._verbose:
            self._logger.info(msg.format(*args))


# Data Loader classes
class DataLoader(ABC):
    def __init__(self, fname, verbose=False):
        self._fname = fname
        self._verbose = verbose
        self._logger = logging.getLogger(self.__class__.__name__)

    @abstractmethod
    def load(self, norm_method='MEAN'):
        """
        Load the data and perform needed preprocessing
        """
        pass

    def log(self, msg, *args):
        if self._verbose:
            self._logger.info(msg.format(*args))


class WineDataLoader(DataLoader):
    def __init__(self, verbose=False):
        # super().__init__('data/winemag-data-small.csv', verbose)
        super().__init__('data/winemag-data-130k-v2.csv', verbose)

    def load(self, norm_method='MEAN'):
        self.log("Loading wine data...")
        df = pd.read_csv(self._fname)

        # Drop NA
        df = df.dropna()

        # clean first column (nonsense column)
        df = df.iloc[:, 1:]

        # Drop unused column
        df = df.drop(columns=['description', 'region_2', 'taster_name', 'taster_twitter_handle', 'title', 'country', 'designation'])

        # Rename column
        df.rename(columns={'region_1':'region'}, inplace=True)

        # Covert points to label
        df = df.assign(
            level=pd.cut(df['points'],
            bins=[80, 86, 88, 91, 100],
            labels=['D', 'C', 'B', 'A'])
        )
        df['level'] = df.level.astype(str)
        df = df.drop(columns=['points'])

        # Encoding data
        encoder_map = {}
        for key in df.columns:
            if key == 'price':
                continue
            encoder_map[key] = preprocessing.LabelEncoder()
            encoder = encoder_map[key]
            data_col = df[key]
            encoder.fit(data_col)
            df[key] = encoder.transform(data_col)

        # Normalized
        for key in df.columns:
            if key == 'level':
                continue
            df[key] = util.normalized(df[key], method=norm_method)

        # print data info
        self.log("Columns: {}", df.columns)

        return df
