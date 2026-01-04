import tensorflow as tf
keras = tf.keras
from . import *
from modules import controller, monitor, logger
from . import _layers as layers
from ._metrics import masked_softmax_cross_entropy, masked_accuracy
import operator
import json

def add_subparser_args(parser):
    subparser = parser.add_argument_group("H2GCN Model Arguments")
    subparser.add_argument("--network_setup", type=str, default="M64-R-T1-G-V-T2-G-V-C1-C2-D0.5-MO")
    subparser.add_argument("--dropout", type=float, default=0.5)
    subparser.add_argument("--hidden", type=int, default=64)
    subparser.add_argument("--adj_nhood", default=["1","2"], type=str, nargs="+")
    subparser.add_argument("--optimizer", type=str, default="adam")
    subparser.add_argument("--lr", type=float, default=0.01)
    subparser.add_argument("--l2_regularize_weight", type=float, default=5e-4)
    subparser.add_argument("--early_stopping", type=int, default=0)
    subparser.add_argument("--best_val_criteria", choices=["val_acc","val_loss"], default="val_acc")
    subparser.add_argument("--save_activations", action="store_true")
    subparser.add_argument("--save_predictions", nargs="+", type=bool, default=True)
    subparser.add_argument("--no_feature_normalize", action="store_true")
    parser.function_hooks["argparse"].append(argparse_callback)

def argparse_callback(args):
    dataset = args.objects["dataset"]
    # Parse network setup string into layer configurations
    layer_setups = parse_network_setup(args.network_setup, dataset.num_labels,
        _dense_units=args.hidden, _dropout_rate=args.dropout,
        parse_preprocessing=True)

    from_layer_types = set([x[0] for x in layer_setups])
    if (Layer.GCN in from_layer_types):
        preprocessing_data(args, getAdjNormHops=args.adj_nhood)
    else:
        preprocessing_data(args, getAdjHops=args.adj_nhood)

    initialize_model(args, layer_setups, args.optimizer, args.lr,
                     args.l2_regularize_weight, args.early_stopping)

def preprocessing_data(args, **kwargs):
    dataset = args.objects["dataset"]
    if not args.no_feature_normalize:
        dataset.row_normalize_features()
    dataset.adj_remove_eye()
    args.objects["tensors"] = vars(dataset.getTensors(getDenseAdj=False, **kwargs))

def initialize_model(args, layer_setups, optimizer, lr,
                     l2_regularize_weight, early_stopping):

    model = H2GCN(layer_setups, l2_regularize_weight=l2_regularize_weight)
    optimizer = keras.optimizers.Adam(learning_rate=lr)
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)

    def train_step(adj, adj_hops, features, y_train, train_mask, **kwargs):
        with tf.GradientTape() as tape:
            preds = model(adj, features, adj_hops, training=True)
            loss = model._loss(preds, y_train, train_mask)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return dict(train_loss=loss)

    def test_step(adj, adj_hops, features, y_train, train_mask,
                  y_val, val_mask, y_test, test_mask,
                  verbose=False, save_activations=False, save_predictions=False, **kwargs):
        preds = model(adj, features, adj_hops, training=False)
        return dict(
            train_acc=masked_accuracy(preds, y_train, train_mask),
            val_acc=masked_accuracy(preds, y_val, val_mask),
            test_accuracy=masked_accuracy(preds, y_test, test_mask),
            val_loss=model._loss(preds, y_val, val_mask),
            test_loss=masked_softmax_cross_entropy(preds, y_test, test_mask),
            monitor=dict()
        )

    # --- Restore Logger and Callbacks ---
    statsPrinter = logger.EpochStatsPrinter()

    def post_epoch_callback(epoch, args):
        # Print stats at the end of each epoch
        epoch_stats_dict = args.objects["epoch_stats"]
        statsPrinter(epoch, epoch_stats_dict)

    # Register objects for the experiment runner
    args.objects["model"] = model
    args.objects["optimizer"] = optimizer
    args.objects["checkpoint"] = checkpoint
    args.objects["train_step"] = train_step
    args.objects["test_step"] = test_step

    # Add logging callback
    args.objects["post_epoch_callbacks"].append(post_epoch_callback)


class H2GCN(tf.keras.Model):
    def __init__(self, layer_setups, sparse_input=True, l2_regularize_weight=0):
        super().__init__()
        self.layer_objs = []
        self.dropout_inds = []
        self.attention_inds = []
        self.supervised_inds = []
        self.graph_inds = []
        self.graph_hops_inds = []
        self.concat_inds = []
        self.experimental_inds = []
        self.embedding_ind = None
        self.output_ind = None
        self.tagsDict = {}

        # --- Build Layers ---
        for ind, (layerType, conf) in enumerate(layer_setups):
            tag = conf.pop("tag", None)

            if layerType == Layer.DENSE:
                layer_cls = layers.SparseDense if sparse_input else keras.layers.Dense
                self.layer_objs.append(
                    layer_cls(conf["units"],
                        kernel_regularizer=keras.regularizers.l2(l2_regularize_weight)
                    )
                )
                if sparse_input: sparse_input = False

            elif layerType == Layer.DROPOUT:
                layer_cls = layers.SparseDropout if sparse_input else keras.layers.Dropout
                self.layer_objs.append(layer_cls(conf["dropout_rate"]))

            elif layerType == Layer.GCN:
                self.graph_hops_inds.append(len(self.layer_objs))
                self.layer_objs.append(layers.GCNLayer(**conf))

            elif layerType == Layer.IDENTITY:
                self.layer_objs.append(tf.sparse.to_dense)
                sparse_input = False

            elif layerType == Layer.RELU:
                self.layer_objs.append(keras.layers.ReLU())

            elif layerType == Layer.VECTORIZE:
                self.layer_objs.append(keras.layers.Flatten())

            elif layerType == Layer.CONCAT:
                self.concat_inds.append(len(self.layer_objs))
                self.layer_objs.append(
                    layers.ConcatLayer(tags=conf["tags"], addInputs=conf["addInputs"])
                )

            if tag:
                self.tagsDict[len(self.layer_objs)-1] = tag

    def call(self, adj, inputs, adjhops, training=False,
             returnBefore=0, executeAfter=0,
             addSupervision=False, saveActivations=None, **kwargs):

        supervisedOutputs = []
        tagged = {}

        if returnBefore <= 0: returnBefore = len(self.layer_objs) + returnBefore
        if executeAfter < 0: executeAfter = len(self.layer_objs) + executeAfter

        for ind, layer in enumerate(self.layer_objs):
            if ind == returnBefore: return inputs
            elif ind < executeAfter: continue

            if ind in self.concat_inds:
                inputs = layer(inputs, **tagged)
            elif ind in self.graph_hops_inds:
                inputs = layer(adjhops, inputs)
            else:
                inputs = layer(inputs)

            if ind in self.tagsDict:
                tagged[self.tagsDict[ind]] = inputs

        return inputs

    def _loss(self, predictions, labels, mask):
        reg = tf.math.add_n(self.losses) if self.losses else 0
        pred = masked_softmax_cross_entropy(predictions, labels, mask)
        return pred + reg
