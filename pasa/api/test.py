from ..utils.io_utils import say
from ..driver.driver import Driver


def main(argv):
    driver = Driver(argv)
    tester = driver.build_tester()
    tester.setup_experiment()
    say('\n\nPREDICTING\n')
    tester.predict()

    if argv.output == 'pretrain':
        tester.output_pretrained_reps()
