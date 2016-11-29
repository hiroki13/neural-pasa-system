from driver import Driver
from ..utils.io_utils import say


def main(argv):
    driver = Driver(argv)
    trainer = driver.build_trainer()
    trainer.setup_experiment()
    say('\n\nTRAINING A MODEL\n')
    trainer.train()
