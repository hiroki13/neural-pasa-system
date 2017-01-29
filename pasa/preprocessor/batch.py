from numpy.random import shuffle
from abc import ABCMeta, abstractmethod


class Batch(object):
    __metaclass__ = ABCMeta

    def __init__(self, batch_size, samples, n_inputs=None):
        self.batch_size = batch_size
        self.n_inputs = len(samples[0].x) + 1 if n_inputs is None else n_inputs
        self.samples = samples
        self.batches = self._set_batches()

    def size(self):
        return len(self.batches)

    def batch_creating_template(self,
                                input_vals,
                                preprocess,
                                get_prev_elems,
                                get_elems,
                                add_input_to_batch):
        batches = []
        batch = [[] for i in xrange(self.n_inputs)]
        input_vals = preprocess(input_vals)
        prev_elems = get_prev_elems(input_vals)

        for input_val in input_vals:
            elems = get_elems(input_val)
            n_samples = len(batch[-1])

            if self._is_batch_boundary(elems, prev_elems, n_samples):
                prev_elems = elems
                batches.append(batch)
                batch = [[] for i in xrange(self.n_inputs)]
            batch = add_input_to_batch(batch, input_val)

        if len(batch[0]) > 0:
            batches.append(batch)

        shuffle(batches)
        return batches

    def _set_batches(self):
        return self.batch_creating_template(input_vals=self.samples,
                                            preprocess=self._preprocess_samples,
                                            get_prev_elems=self._get_prev_elems_samples,
                                            get_elems=self._get_elems_samples,
                                            add_input_to_batch=self._add_sample_to_batch)

    def shuffle_batches(self):
        self.batches = self.batch_creating_template(input_vals=self.batches,
                                                    preprocess=self._preprocess_batches,
                                                    get_prev_elems=self._get_prev_elems_batches,
                                                    get_elems=self._get_elems_batches,
                                                    add_input_to_batch=self._add_input_to_batch)

    @abstractmethod
    def _preprocess_samples(self, samples):
        raise NotImplementedError()

    @abstractmethod
    def _preprocess_batches(self, batches):
        raise NotImplementedError()

    @abstractmethod
    def _get_prev_elems_samples(self, samples):
        raise NotImplementedError()

    @abstractmethod
    def _get_prev_elems_batches(self, input_vals):
        raise NotImplementedError()

    @abstractmethod
    def _get_elems_samples(self, sample):
        raise NotImplementedError()

    @abstractmethod
    def _get_elems_batches(self, input_val):
        raise NotImplementedError()

    @abstractmethod
    def _is_batch_boundary(self, elems, prev_elems, n_samples):
        raise NotImplementedError()

    @abstractmethod
    def _add_sample_to_batch(self, batch, sample):
        raise NotImplementedError

    @abstractmethod
    def _add_input_to_batch(self, batch, input_val):
        raise NotImplementedError

    @staticmethod
    def _extract_samples_including_prds(samples):
        return [sample for sample in samples if sample.n_prds > 0]

    @staticmethod
    def _separate_samples(samples):
        return [elem for sample in samples for elem in zip(*sample.inputs)]

    @staticmethod
    def _separate_batches(batches):
        return [elem for batch in batches for elem in zip(*batch)]


class BaseBatch(Batch):

    def _preprocess_samples(self, samples):
        samples = self._extract_samples_including_prds(samples)
        return self._sort_samples(samples)

    def _preprocess_batches(self, batches):
        input_vals = self._separate_batches(batches)
        return self._sort_input_vals(input_vals)

    def _get_prev_elems_samples(self, samples):
        return [samples[0].n_words]

    def _get_prev_elems_batches(self, input_vals):
        return [len(input_vals[0][0])]

    def _get_elems_samples(self, sample):
        return [sample.n_words]

    def _get_elems_batches(self, input_val):
        return [len(input_val[0])]

    @staticmethod
    def _sort_samples(samples):
        shuffle(samples)
        samples.sort(key=lambda sample: sample.n_words)
        return samples

    @staticmethod
    def _sort_input_vals(inputs):
        shuffle(inputs)
        inputs.sort(key=lambda elem: len(elem[0]))
        return inputs

    def _is_batch_boundary(self, elems, prev_elems, n_samples):
        n_words = elems[0]
        prev_n_words = prev_elems[0]
        if prev_n_words != n_words or n_samples >= self.batch_size:
            return True
        return False

    def _add_sample_to_batch(self, batch, sample):
        inputs = sample.x + [sample.y]
        for i, elem in enumerate(inputs):
            batch[i].extend(elem)
        return batch

    def _add_input_to_batch(self, batch, input_val):
        for i, elem in enumerate(input_val):
            batch[i].append(elem)
        return batch


class GridBatch(BaseBatch):

    def _preprocess_batches(self, batches):
        input_vals = self._separate_batches(batches)
        return self._sort_input_vals(input_vals)

    def _get_prev_elems_samples(self, samples):
        return [samples[0].n_prds, samples[0].n_words]

    def _get_elems_samples(self, sample):
        return [sample.n_prds, sample.n_words]

    def _get_prev_elems_batches(self, input_vals):
        n_prds = len(input_vals[0][0])
        n_words = len(input_vals[0][0][0])
        return [n_prds, n_words]

    def _get_elems_batches(self, input_val):
        n_prds = len(input_val[0])
        n_words = len(input_val[0][0])
        return [n_prds, n_words]

    @staticmethod
    def _sort_samples(samples):
        shuffle(samples)
        samples.sort(key=lambda sample: sample.n_prds)
        samples.sort(key=lambda sample: sample.n_words)
        return samples

    @staticmethod
    def _sort_input_vals(inputs):
        shuffle(inputs)
        inputs.sort(key=lambda elem: len(elem[0]))  # n_prds
        inputs.sort(key=lambda elem: len(elem[0][0]))  # n_words
        return inputs

    def _is_batch_boundary(self, elems, prev_elems, n_samples):
        n_prds, n_words = elems
        prev_n_prds, prev_n_words = prev_elems
        if prev_n_words != n_words or n_prds != prev_n_prds or n_samples >= self.batch_size:
            return True
        return False

    def _add_sample_to_batch(self, batch, sample):
        inputs = sample.x + [sample.y]
        for i, elem in enumerate(inputs):
            batch[i].append(elem)
        return batch

    def _add_input_to_batch(self, batch, input_val):
        for i, elem in enumerate(input_val):
            batch[i].append(elem)
        return batch
