"""A quick example of using weka's python wrappers..."""
import os.path as op
from sklearn.datasets import load_iris
from biores.models.sklearn_utils import skl_factory
import numpy as np
from oscail.common.integration.weka_classifiers_wrappers import WCNaiveBayes, WCRandomForest
from oscail.common.evaluation import ResultWriter, cross_validate, ResultReader, CVer
from oscail.common.examples import BasicExamplesProvider, join_maybetoolong_fn
from oscail.common.integration.weka_utils import weka_object_to_oscail


def iris_data():
    """Returns a copy of the mighty Fisher iris dataset."""
    X, y = load_iris()['data'], load_iris()['target']
    y[y == 2.] = 0  # N.B. make binary, TODO simulate a competition dataset
    return BasicExamplesProvider(X, y)


def auc_for_result(result_dir):
    """Reads the AUC from a the directory of a result.
    Returns mean AUC and standard deviation.
    """
    rr = ResultReader(result_dir)
    aucs = [rr.auc_from_fn(fold) for fold in rr.present_folds()]
    return np.mean(aucs), np.std(aucs)


def evaluate_model(model,
                   data_provider=None, cvseed=0, num_folds=10,
                   dest=op.join(op.expanduser('~'), 'weka-exp')):
    """An example of a generic function evaluating a model on a dataset.

    Evaluation and models are stored on disk for later analysis and blending,
    keeping full provenance information and allowing evaluation restart.

    Returns the directory in which the experiment results are stored.
    """

    if data_provider is None:
        data_provider = iris_data()

    # An object to provide disjoint partitions of the dataset between train and test
    cver = CVer(data_provider.X(), data_provider.Y(),
                num_folds=num_folds, seed=cvseed, stratify=True)

    # The destination dir will depend on the model setup
    dest = join_maybetoolong_fn(dest, model.configuration.id())

    # An object to write partial cross-validation results as they are done
    rw = ResultWriter(fp=data_provider,
                      cver=cver,
                      classifier=model,  # Here we do not know which model...
                      root=dest,
                      save_model_setup=False)

    # Cross validate the model
    cross_validate(rw, num_processes=1, reraise=False, should_copy=False,
                   calibrate=False)

    return dest


if __name__ == '__main__':

    # Evaluate a model in python-land
    skl_nb = skl_factory('gnb')
    skl_nb_result_dir = evaluate_model(skl_nb)

    # Evaluate a model in java-land
    weka_nb = WCNaiveBayes(useSupervisedDiscretization=True)
    weka_nb_result_dir = evaluate_model(weka_object_to_oscail(weka_nb))

    # Evaluate a model in java-land, but with different configuration
    weka_nb = WCNaiveBayes(useSupervisedDiscretization=False)
    weka_nb_result_dir2 = evaluate_model(weka_object_to_oscail(weka_nb))

    rf = WCRandomForest(maxDepth=100)
    rf_dest = evaluate_model(weka_object_to_oscail(rf))

    # How did they do?
    print('-' * 80)
    print('ScikitNB AUC: %.4f +/- %.4f' % auc_for_result(skl_nb_result_dir))
    print('WekaNB AUC: %.4f +/- %.4f' % auc_for_result(weka_nb_result_dir))
    print('WekaNB2 AUC: %.4f +/- %.4f' % auc_for_result(weka_nb_result_dir2))
    print('RF AUC: %.4f +/- %.4f' % auc_for_result(rf_dest))
    print('-' * 80)

    #nb = WCNaiveBayes(useSupervisedDiscretization=True)
    #print 'nb is an object in "java-land"', type(nb)
    #print nb.getOptions()
    #print nb.listOptions()
    #print nb.buildClassifier
    #print nb.configuration
    #print nb.configuration.configuration_string()
    #print nb.configuration.id()

    print('Done')
    exit(0)