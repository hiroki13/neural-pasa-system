from ..experimenter.trainer import *
from ..experimenter.tester import *
from ..experimenter.epoch_manager import *
from ..preprocessor.preprocessor import *
from ..model.model_api import *

from ..utils.io_utils import load_data


class Driver(object):

    def __init__(self, argv):
        self.argv = argv

    def build_trainer(self):
        argv = self.argv
        trainer = self._select_trainer(argv)
        preprocessor = self._select_preprocessor(argv)
        model_api = self._select_model_api(argv)
        epoch_manager = self._select_epoch_manager(argv)
        return trainer(argv=argv,
                       preprocessor=preprocessor(argv),
                       model_api=model_api(argv),
                       epoch_manager=epoch_manager(argv))

    def build_tester(self):
        argv = self.argv
        tester = self._select_tester(argv)
        preprocessor = self._select_preprocessor(argv)
        model_api = self._select_model_api(argv)
        config = self._load_config(argv)
        return tester(argv=argv,
                      preprocessor=preprocessor(argv, config),
                      model_api=model_api(config),
                      config=config)

    @staticmethod
    def _select_trainer(argv):
        return Trainer

    @staticmethod
    def _select_tester(argv):
        return Tester

    @staticmethod
    def _select_preprocessor(argv):
        if argv.model == 'inter':
            return InterPreprocessor
        return BasePreprocessor

    @staticmethod
    def _select_model_api(argv):
        if argv.model == 'grid':
            return GridModelAPI
        elif argv.model == 'inter':
            return MentionPairModelAPI
        return BaseModelAPI

    @staticmethod
    def _select_epoch_manager(argv):
        return EpochManager

    @staticmethod
    def _load_config(argv):
        config = load_data(argv.load_config)
        config.n_best = argv.n_best
        return config
