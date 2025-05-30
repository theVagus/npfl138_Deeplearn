#!/usr/bin/env python3
import argparse
import datetime
import os
import re

import torch

import npfl138
npfl138.require_version("2425.8")
from npfl138.datasets.morpho_dataset import MorphoDataset

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--batch_size", default=10, type=int, help="Batch size.")
parser.add_argument("--epochs", default=5, type=int, help="Number of epochs.")
parser.add_argument("--max_sentences", default=None, type=int, help="Maximum number of sentences to load.")
parser.add_argument("--recodex", default=False, action="store_true", help="Evaluation in ReCodEx.")
parser.add_argument("--rnn", default="LSTM", choices=["LSTM", "GRU"], help="RNN layer type.")
parser.add_argument("--rnn_dim", default=64, type=int, help="RNN layer dimension.")
parser.add_argument("--seed", default=41, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
parser.add_argument("--we_dim", default=128, type=int, help="Word embedding dimension.")
# If you add more arguments, ReCodEx will keep them with your default values.


class Model(npfl138.TrainableModule):
    def __init__(self, args: argparse.Namespace, train: MorphoDataset.Dataset) -> None:
        super().__init__()
        # Create all needed layers.
        # TODO(tagger_we): Create a `torch.nn.Embedding` layer, embedding the word ids
        # from `train.words.string_vocab` to dimensionality `args.we_dim`.
        self._word_embedding = ...

        # TODO(tagger_we): Create an RNN layer, either `torch.nn.LSTM` or `torch.nn.GRU` depending
        # on `args.rnn`. The layer should be bidirectional (`bidirectional=True`) with
        # dimensionality `args.rnn_dim`. During the model computation, the layer will
        # process the word embeddings generated by the `self._word_embedding` layer,
        # and we will sum the outputs of forward and backward directions.
        self._word_rnn = ...

        # TODO(tagger_we): Create an output linear layer (`torch.nn.Linear`) processing the RNN output,
        # producing logits for tag prediction; `train.tags.string_vocab` is the tag vocabulary.
        self._output_layer = ...

    def forward(self, word_ids: torch.Tensor) -> torch.Tensor:
        # TODO(tagger_we): Start by embedding the `word_ids` using the word embedding layer.
        hidden = ...

        # TODO(tagger_we): Process the embedded words through the RNN layer. Because the sentences
        # have different length, you have to use `torch.nn.utils.rnn.pack_padded_sequence`
        # to construct a variable-length `PackedSequence` from the input. You need to compute
        # the length of each sentence in the batch (by counting non-`MorphoDataset.PAD` tokens);
        # note that these lengths must be on CPU, so you might need to use the `.cpu()` method.
        # Finally, also pass `batch_first=True` and `enforce_sorted=False` to the call.
        packed = ...

        # TODO(tagger_we): Pass the `PackedSequence` through the RNN, choosing the appropriate output.
        packed = ...

        # TODO(tagger_we): Unpack the RNN output using the `torch.nn.utils.rnn.pad_packed_sequence` with
        # `batch_first=True` argument. Then sum the outputs of forward and backward directions.
        hidden = ...

        # TODO(tagger_we): Pass the RNN output through the output layer. Such an output has a shape
        # `[batch_size, sequence_length, num_tags]`, but the loss and the metric expect
        # the `num_tags` dimension to be in front (`[batch_size, num_tags, sequence_length]`),
        # so you need to reorder the dimensions.
        hidden = ...

        return hidden

    def compute_loss(self, y_pred: torch.Tensor, y_true: torch.Tensor, word_ids: torch.Tensor) -> torch.Tensor:
        # TODO: Compute the loss as the negative log-likelihood of the gold data `y_true`.
        # The computation must process the whole batch at once.
        # - Start by computing the log probabilities of the predictions using the `log_softmax` method.
        # - Compute the alphas according to the CTC algorithm.
        # - Then, you need to take the appropriate alpha for every batch example (using the corresponding
        #   lengths of `y_pred` and also `y_true`) and compute the loss from it.
        # - The losses of the individual examples should be divided by the length of the
        #   target sequence (excluding padding; use 1 if the target sequence is empty).
        #   - This is because we want to compute averages per logit; the `torch.nn.CTCLoss` does
        #     exactly the same when `reduction="mean"` (the default) is used.
        # - Finally, return the mean of the resulting losses.
        #
        # Several comments:
        # - You can add numbers represented in log space using `torch.logsumexp`/`torch.logaddexp`.
        # - With a slight abuse of notation, use `MorphoDataset.PAD` as the blank label in the CTC algorithm
        #   because `MorphoDataset.PAD` is never a valid output tag.
        # - During the computation, I use `-1e9` as the representation of negative infinity; using
        #   `-torch.inf` did not work for me because some operations were not well defined.
        # - During the loss computation, in some cases the target sequence cannot be produced at all.
        #   In that case return 0 as the loss (the same behaviour as passing `zero_infinity=True`
        #   to `torch.nn.CTCLoss`).
        # - During development, you can compare your outputs to the outputs of `torch.nn.CTCLoss`
        #   (with `reduction="none"` you can compare individual batch examples; in that case,
        #   the normalization by the target sequence lengths is not performed).  However, in ReCodEx,
        #   `torch.nn.CTCLoss` is not available.
        raise NotImplementedError()

    def ctc_decoding(self, logits: torch.Tensor, word_ids: torch.Tensor) -> list[torch.Tensor]:
        # TODO: Implement greedy CTC decoding algorithm. The result should be a list of
        # decoded tag sequences, each sequence (batch example) with appropriate length
        # (i.e., at this point we do not pad the predictions in the batch to the same length).
        #
        # The greedy algorithm should, for every batch example:
        # - consider only the predictions corresponding to valid words (i.e., not the padding ones);
        # - compute the most probable extended label for every one of them;
        # - remove repeated labels;
        # - finally remove the blank labels (which are `MorphoDataset.PAD` in our case).
        raise NotImplementedError()

    def compute_metrics(
        self, y_pred: torch.Tensor, y_true: torch.Tensor, word_ids: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        # TODO: Compute predictions using the `ctc_decoding`.
        predictions = ...

        self.metrics["edit_distance"].update(predictions, y_true)
        return {name: metric.compute() for name, metric in self.metrics.items()}

    def predict_step(self, xs, as_numpy=True):
        with torch.no_grad():
            # Perform constrained decoding.
            batch = self.ctc_decoding(self.forward(*xs), *xs)
            if as_numpy:
                batch = [example.numpy(force=True) for example in batch]
            return batch


class TrainableDataset(npfl138.TransformedDataset):
    def transform(self, example):
        # TODO(tagger_we): Construct a single example, each consisting of the following pair:
        # - a PyTorch tensor of integer ids of input words as input,
        # - a PyTorch tensor of integer tag ids as targets.
        # To create the ids, use `string_vocab` of `self.dataset.words` and `self.dataset.tags`.
        #
        # TODO: However, compared to `tagger_we`, keep in the target sequence only the tags
        # starting with "B-" (before remapping them to ids).
        word_ids = ...
        tag_ids = ...
        return word_ids, tag_ids

    def collate(self, batch):
        # Construct a single batch, where `batch` is a list of examples
        # generated by `transform`.
        word_ids, tag_ids = zip(*batch)
        # TODO(tagger_we): Combine `word_ids` into a single tensor, padding shorter
        # sequences to length of the longest sequence in the batch with zeros
        # using `torch.nn.utils.rnn.pad_sequence` with `batch_first=True` argument.
        word_ids = ...
        # TODO(tagger_we): Process `tag_ids` analogously to `word_ids`.
        tag_ids = ...
        return word_ids, tag_ids


def main(args: argparse.Namespace) -> dict[str, float]:
    # Set the random seed and the number of threads.
    npfl138.startup(args.seed, args.threads)
    npfl138.global_keras_initializers()

    # Create logdir name.
    args.logdir = os.path.join("logs", "{}-{}-{}".format(
        os.path.basename(globals().get("__file__", "notebook")),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", k), v) for k, v in sorted(vars(args).items())))
    ))

    # Load the data.
    morpho = MorphoDataset("czech_cnec", max_sentences=args.max_sentences)

    # Prepare the data for training.
    train = TrainableDataset(morpho.train).dataloader(batch_size=args.batch_size, shuffle=True)
    dev = TrainableDataset(morpho.dev).dataloader(batch_size=args.batch_size)

    # Create the model and train.
    model = Model(args, morpho.train)

    model.configure(
        # TODO(tagger_we): Create the Adam optimizer.
        optimizer=...,
        # We compute the loss using `compute_loss` method, so no `loss` is passed here.
        metrics={
            # TODO: Create `npfl138.metrics.EditDistance` evaluating CTC greedy decoding, passing
            # `ignore_index=morpho.PAD`.
            "edit_distance": ...,
        },
        logdir=args.logdir,
    )

    logs = model.fit(train, dev=dev, epochs=args.epochs)

    # Return all metrics for ReCodEx to validate.
    return {metric: value for metric, value in logs.items()}


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    main(main_args)
