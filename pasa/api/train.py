from ..utils.io_utils import say
from ..driver.driver import Driver


def main(argv):
    driver = Driver(argv)
    trainer = driver.build_trainer()
    trainer.setup_experiment()
    say('\n\nTRAINING A MODEL\n')
    trainer.train()
