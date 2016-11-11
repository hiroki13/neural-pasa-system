from ..experimenter.experimenter import *
from ..experimenter.epoch_manager import *
from ..preprocessor.preprocessor import *
from ..model.model_api import *


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
        if argv.model == 'jack':
            return JackKnifeTrainer
        elif argv.model == 'sep':
            return TrainCorpusSeparator
        elif argv.model == 'stack':
            return StackingTrainer
        elif argv.model == 'rerank':
            return RerankingTrainer
        return Trainer

    @staticmethod
    def _select_tester(argv):
        if argv.model == 'nbest':
            return NBestTester
        elif argv.model == 'jack':
            return JackKnifeTester
        return Tester

    @staticmethod
    def _select_preprocessor(argv):
        if argv.model == 'rerank':
            return RerankingPreprocessor
        elif argv.model == 'stack':
            return StackingPreprocessor
        elif argv.model == 'grid':
            return GridPreprocessor
        return Preprocessor

    @staticmethod
    def _select_model_api(argv):
        if argv.model == 'nbest':
            return NBestModelAPI
        elif argv.model == 'stack':
            return StackingModelAPI
        elif argv.model == 'rerank':
            return RerankingModelAPI
        elif argv.model == 'grid':
            return GridModelAPI
        elif argv.model == 'jack':
            if argv.output == 'n_best':
                return NBestModelAPI
        return ModelAPI

    @staticmethod
    def _select_epoch_manager(argv):
        return EpochManager

    @staticmethod
    def _load_config(argv):
        config = load_data(argv.load_config)
        config.n_best = argv.n_best
        return config
