# -*- coding: utf-8 -*-
"""
Calling weka stuff from python using JPype, a deference to Flo. See:
  - http://kogs-www.informatik.uni-hamburg.de/~meine/weka-python/
  - http://iacobelli.cl/blog/?p=119
  - http://dimitri-christodoulou.blogspot.com.es/2012/03/use-weka-in-your-python-code.html

We use good old JPype, which is getting a bit of a facelift here:
  - https://pypi.python.org/pypi/JPype1

Other options: Py4J, JCC, Pyjnius, javabridge(https://github.com/CellProfiler/python-javabridge)
"""
import os
import os.path as op

import numpy as np

from jpype import *

from oscail.common.integration.jpype_utils import java_class, discover_subclasses, get_class_for, jpype_bootup
from oscail.common.configuration import Configurable, Configuration

jpype_bootup()  # On import, initialize the java JVM

#################################
# Java imports
#################################
#Core
Instances = JClass('weka.core.Instances')
Instance = JClass('weka.core.Instance')
Attribute = JClass('weka.core.Attribute')
OptionHandler = JClass('weka.core.OptionHandler')
DenseInstance = JClass('weka.core.DenseInstance')
#Classifiers
Classifier = JClass('weka.classifiers.Classifier')
Evaluation = JClass('weka.classifiers.Evaluation')
#Filters
NominalToBinary = JClass('weka.filters.unsupervised.attribute.NominalToBinary')
Filter = JClass('weka.filters.Filter')
#FeatureSelection
AS = JPackage('weka.attributeSelection')
#Java
BufferedReader = JClass('java.io.BufferedReader')
FileReader = JClass('java.io.FileReader')
#Bean introspection
Introspector = JClass('java.beans.Introspector')


##################################
# Data management
# The big challenge here is transfering from python to java and back
# big chunks of data. It is slow. When possible, do it at once and/or
# use indexing to minimize data transfer.
##################################
def read_arff_into_weka(arff_file, class_index=-1):
    """Loads data from an ARFF directly in java land.
    This method is as efficient as it gets, with minimum exchange between python and java land.

    Parameters:
      - arff_file: the path to the arff file to load
      - class_index: the index of the label attribute (<0 means last attribute, as per weka conventions)

    Returns: a handle to the Instances object with the data in java land.
    """
    # TODO: It is easy to extend to allow loading from other sources.
    reader = BufferedReader(FileReader(arff_file))
    instances = Instances(reader)
    reader.close()
    if not class_index is None:
        if class_index < 0:
            class_index = instances.numAttributes() - 1
        instances.setClassIndex(class_index)
    return instances


def add_method_from_instances(arity=1):
    """Returns the add method from weka Instances class.
    Given weka's API:
     http://weka.sourceforge.net/doc.dev/weka/core/Instances.html
    arity 1 means retrieving the method add(Instance), while arity 2 retrieves add(int, Instance)
    """

    ######
    #We need this method, unfortunatelly, because with newer weka versions,
    #in which Instances is generic, this fails:
    #  instances = Instances()
    #  instances.add(instance)
    #See http://sourceforge.net/mailarchive/forum.php?forum_name=jpype-users&max_rows=25&style=nested&viewmonth=200704
    #
    #Therefore we need some reflection to workaround old-unmaintained jpype bugs caused by
    #the lame java generics implementation.
    ######

    if not 0 < arity < 3:
        raise Exception('Instances only have methods called "add" of arity 1 and 2')

    def is_desired_instances_add_method(m):
        return m.getName() == 'add' and len(m.getParameterTypes()) == arity  # TODO: error prone on API change,
                                                                             # check the whole signature
    return next(m for m in reflect.getDeclaredMethods(Instances) if is_desired_instances_add_method(m))


def select_instances(instances, indices):
    """Creates a new weka Instances object with the same schema as instances and populates it with the
    examples in indices, that should be a list or an array of integers between 0 and num_instances.
    This allows as little transference of data between python and java lands.
    """
    instances_sample = Instances(instances, 0)
    add_method = add_method_from_instances(arity=1)
    array_class = get_class_for('java.lang.Object[]')
    for i in indices:
        add_method.invoke(instances_sample, array_class([instances.get(int(i))]))
    return instances_sample


def numpy2java(data, to_list=True):
    """Copies a numpy array (data) into a (jagged) double array in java land and returns a handle to the java object.

    to_list indicates whether to optimize for speed:
      - if True, numpy transforms the data into python objects as requested by jpype,
        but this ensures there will eventually be 3 copies of the data in memory)
      - if False, no data duplication happens in python-land, but the conversion
        to python doubles is potentially slower
    Need to benchmark this...

    There should be a way of doing this much faster using, for example, python's buffer protocol
    and java's NIO. Pyjnius does not support it either AFAIK.
    """
    if to_list:
        return JArray(JDouble, data.ndim)(data.tolist())
    return JArray(JDouble, data.ndim)(data)


def numpy2instances(X, y, regression=False, to_list=True, verbose=False):
    """Copies a numpy array of features X and the actual outputs (y)
    into a weka's Instances object in java-land.

    Caveats:
      - at the moment only binary classification or regression labels are accepted
      - this is very slow, and, of course, it duplicates the dataset in memory.
        In most cases it would be much more efficient to load the data from disk directly in java-land

    Parameters:
      - X: a num_examples x num_features numpy array with the examples
      - y: a num_examples array of labels (regression or binary classification in {0,1})
      - regression: are the labels continuous (regression) or nominal (classification)?
      - to_list: see docstring of numpy2java
      - verbose: dump some log messages to stdout

    Returns: a handle to an instances object in java land containing X,y as required by weka
    """

    if verbose:
        print 'Moving data from python to java'

    #1 Generate schema
    def generate_schema():
        attributes = JClass('java.util.ArrayList')()
        for i in range(X.shape[1]):
            attributes.add(Attribute('D%d' % (i+1)))
        if regression:
            attributes.add(Attribute('Activity'))
        else:  # Assume binary class TODO: allow multiclass too!
            nominal_values = JClass('java.util.ArrayList')()
            nominal_values.add('0')
            nominal_values.add('1')
            attributes.add(Attribute('Activity', nominal_values))
        schema = Instances('from_python_x,y', attributes, X.shape[0])
        schema.setClassIndex(X.shape[1])
        return schema

    #2 Populate (the bottleneck)
    def populate_instances(schema):
        # Numpy to java double []
        xy = np.hstack((X, y[:, np.newaxis]))
        data = numpy2java(xy, to_list=to_list)
        # Introspection to get Instances add method and the class of an array of objects in java land
        add_method = add_method_from_instances(arity=1)
        array_class = get_class_for('java.lang.Object[]')
        # Now we are ready to instantiate the examples in java-land
        for instance in data:
            add_method.invoke(schema, array_class([DenseInstance(1.0, instance)]))
        if verbose:
            print 'Done moving data from python to java'
        return schema

    return populate_instances(generate_schema())


##################################
# Feature selection
##################################
class WekaFeatureSelection(object):
    """Convenience methods to setup weka feature selectors."""
    @staticmethod
    def asevaluator(evaluator, instances):
        evaluator.buildEvaluator(instances)
        classIndex = instances.classIndex()
        return [evaluator.evaluateAttribute(i) for i in range(instances.numAttributes())
                if i != classIndex]

    @staticmethod
    def infogain(instances):
        return WekaFeatureSelection.asevaluator(AS.InfoGainAttributeEval(), instances)

    @staticmethod
    def chi2(instances):
        return WekaFeatureSelection.asevaluator(AS.ChiSquaredAttributeEval(), instances)

    @staticmethod
    def su(instances):
        return WekaFeatureSelection.asevaluator(AS.SymmetricalUncertAttributeEval(), instances)

    @staticmethod
    def relief(instances):
        return WekaFeatureSelection.asevaluator(AS.ReliefFAttributeEval(), instances)

    @staticmethod
    def gainratio(instances):
        return WekaFeatureSelection.asevaluator(AS.GainRatioAttributeEval(), instances)

    @staticmethod
    def oner(instances):
        return WekaFeatureSelection.asevaluator(AS.OneRAttributeEval(), instances)


##################################
# Wrappers to play well with our simple Configuration framework
##################################

class WekaParameter(object):
    def __init__(self, name, setter, getter, value):  # , java_type=None
        super(WekaParameter, self).__init__()
        self.name = name
        self.setter = setter
        self.getter = getter
        #self.type = java_type   # TODO: java types are not pickable, convert to python types and back
        self.value = value


class WekaWrapper(Configurable):
    """A configurable and lazy container for a weka object.
    The challenge is to properly handle nested stuff like filters or base classifiers.
    Other options: check partial, lazypy.
    """
#TODO: SelectedTag management
#TODO: support Files
#TODO: Array management
#TODO: Copyability and pickability using serializable (prioritary)
#TODO: no_id management (manual, use dictionary)
#TODO: check how they find the properties in weka and use it here
    def __init__(self, weka_class_name, parameters, non_ids=(), suffix=''):
        super(WekaWrapper, self).__init__()
        self.weka_class_name = weka_class_name
        self.weka_parameters = parameters
        self.non_ids = non_ids
        self.suffix = suffix

    @property
    def configuration(self):
        name = self.weka_class_name.rpartition('.')[2]
        config_dict = dict((param.name, param.value) for param in self.weka_parameters)
        return Configuration(name, config_dict, self.non_ids)

    def instantiate(self):
        """Create an instance of the weka object and set it up correctly."""
        weka_object = JClass(self.weka_class_name)()
        #Configure accordingly
        for param in self.weka_parameters:
            v = param.value
            if isinstance(v, WekaWrapper):
                v = v.instantiate()
            if not v is None:
                getattr(weka_object, param.setter)(v)  # Set the value
        return weka_object

    def instantiation_function(self, prefix='WC'):
        """Generate a function to instantiate the class and set it up in a more pythonic way."""
        java_class_name = self.weka_class_name.rpartition('.')[2]
        name = java_class_name.lower() + self.suffix
        indent_size = len('def %s%s%s(' % (prefix, java_class_name, self.suffix))
        parameters_list = ['%s=%s' % (param.name, param.value if not isinstance(param.value, WekaWrapper) else None)
                           for param in self.weka_parameters]
        declaration_st = 'def %s%s%s(%s):' % (prefix, java_class_name, self.suffix,
                                              (',\n' + ' '*indent_size).join(parameters_list))
        indent = ' ' * 4
        instantiation_string = indent + '%s = %s%s()' % (name, java_class_name, self.suffix)
        configuration_list = [indent + 'if not %s is None: %s.%s(%s)' % (param.name, name, param.setter, param.name)
                              for param in self.weka_parameters]
        return_st = indent + 'return %s' % name
        method_st = '\n'.join([declaration_st] + [instantiation_string] + configuration_list + [return_st])
        return method_st

    def manage_weka_optionhandlers(self):
    #Check the OptionHandler interface
    #http://weka.sourceforge.net/doc.dev/weka/core/OptionHandler.html
    #getOptions() / setOptions(String[]) / listOptions()
    #Collect the options
    #opList = weka_object.listOptions()
    #while opList.hasMoreElements():
    #    option = opList.nextElement() #http://weka.sourceforge.net/doc.dev/weka/core/Option.html
    #    print option.name()           #e.g  Q
    #    print option.description()    #e.g. Seed for random data shuffling (default 1).
    #    print option.numArguments()   #e.g. 1
    #    print option.synopsis()       #e.g. -Q <seed>
    #    print option.getRevision()    #e.g. 8034
        pass


class WekaClassifier(WekaWrapper):
    def __init__(self, weka_class_name, parameters, non_ids=(), instances_schema=None):
        super(WekaClassifier, self).__init__(weka_class_name, parameters, non_ids)
        self.instances_schema = instances_schema
        self.classifier = None

    def train(self, x, y=None):
        if isinstance(x, np.ndarray):
            x = numpy2instances(x, y)
        if not isinstance(x, Instances):
            raise Exception('Weka Classifiers only take instances (or numpy arrays) ATM')
        self.classifier = self.instantiate()
        self.classifier.buildClassifier(x)
        return self

    def scores(self, x):
        if isinstance(x, np.ndarray):
            x = numpy2instances(x, np.zeros(len(x)))
        if not isinstance(x, Instances):
            raise Exception('Weka Classifiers only take instances (or numpy arrays) ATM')
        return np.array([self.classifier.distributionForInstance(x.instance(i)) for i in range(x.numInstances())])


def weka_object_to_oscail(weka_object):
    """Instantiates a WekaWrapper using java reflection."""
    TO_IGNORE = {'options', 'revision', 'capabilities', 'class', 'technicalInformation'}
    #A bit weird the way of doing this, but works in a reasonable amount of cases
    bi = Introspector.getBeanInfo(weka_object.getClass())
    pds = bi.getPropertyDescriptors()
    parameters = []
    for pd in pds:
        p_name = pd.getName()
        p_reader = pd.getReadMethod()
        p_writer = pd.getWriteMethod()
        p_type = pd.getPropertyType()
        if p_name in TO_IGNORE or not p_reader or not p_writer:
            continue
        if 'lambda' == p_name:
            p_name = 'lambda_'
        # Lame rules-of-thumb for default, should just allow primitive types
        p_value = eval('weka_object.%s()' % p_reader.getName())
        if 'jpype._jclass.boolean' in str(p_type):  # Is there any better way to check for this?
            p_value = True if p_value else False
        if 'jpype._jclass.java.lang.String' in str(p_type):
            p_value = '\'%s\'' % p_value
        if 'File' in str(p_type):
            p_value = None
        if '[]' in str(p_type):  # Arrays
            p_value = None
        if 'SelectedTag' in str(p_type):  # SelectedTags
            p_value = None
        if isinstance(p_value, OptionHandler):  # OptionHandler
            #p_value=None
            p_value = weka_object_to_oscail(p_value)
        parameters.append(WekaParameter(p_name, p_writer.getName(), p_reader.getName(), p_value))  # java_type=p_type
    my_type = WekaClassifier if isinstance(weka_object, Classifier) else WekaWrapper
    return my_type(java_class(weka_object).getName(), parameters)


################################################
#Generate soure code to allow "pythonic" setup of weka stuff
################################################

def generate_classifier_wrappers():
    available_classifiers = discover_subclasses(Classifier)

    def is_banned(classifier):
        banned = {
            'weka.classifiers.trees.lmt.LMTNode',
            'weka.classifiers.trees.m5.PreConstructedLinearModel',
            'weka.classifiers.trees.m5.RuleNode',
            'weka.classifiers.bayes.NaiveBayesMultinomialText',
            'weka.classifiers.functions.SGDText',
            u'weka.classifiers.meta.RotationForest$ClassifierWrapper',
            'weka.classifiers.trees.ft.FTInnerNode',
            'weka.classifiers.trees.ft.FTLeavesNode',
            'weka.classifiers.trees.ft.FTNode',
            'weka.classifiers.trees.UserClassifier',
            'weka.classifiers.rules.DTNB',
            'weka.classifiers.meta.EnsembleSelection',
        }
        return 'pmml' in classifier or classifier in banned
    weka_wrappers = []
    weka_imports = []
    for i, class_info in enumerate(available_classifiers):
        print 'Class %d of %d: %s' % (i, available_classifiers.size(), class_info.getClassName())
        if not is_banned(class_info.getClassName()):
            weka_imports.append('%s = JClass(\'%s\')' % (class_info.getClassName().rpartition('.')[2],
                                                         class_info.getClassName()))
            weka_wrappers.append(weka_object_to_oscail(JClass(class_info.getClassName())()))
        else:
            print 'ignored'
    dest = op.join(op.dirname(__file__), 'weka_classifiers_wrappers.py')
    if op.exists(dest):
        os.remove(dest)
    with open(dest, 'w') as writer:
        writer.write('from jpype import *\n')
        writer.write('from oscail.common.integration.jpype_utils import jpype_bootup\n\n')
        writer.write('jpype_bootup()\n\n')
        writer.write('\n'.join(weka_imports) + '\n\n')
        writer.write('\n\n'.join(ww.instantiation_function('WC') for ww in weka_wrappers))


def generate_filter_wrappers():
    available_filters = discover_subclasses(Filter)

    def is_banned(wfilter):
        banned = {
            'weka.filters.unsupervised.attribute.Add',
            'weka.filters.unsupervised.attribute.ChangeDateFormat',
            'weka.filters.unsupervised.attribute.StringToWordVector',
        }
        return wfilter in banned

    def suffix_for_filter(cn):
        if '.supervised.attribute.' in cn:
            return '_sa'
        if '.supervised.instance.' in cn:
            return '_si'
        if '.unsupervised.attribute.' in cn:
            return '_ua'
        if '.unsupervised.instance.' in cn:
            return '_ui'
        return ''
    weka_wrappers = []
    weka_imports = []
    for i, class_info in enumerate(available_filters):
        cn = class_info.getClassName()
        print 'Class %d of %d: %s' % (i, available_filters.size(), cn)
        suffix = suffix_for_filter(cn)
        if not is_banned(cn):
            weka_imports.append('%s%s=JClass(\'%s\')' % (cn.rpartition('.')[2], suffix, class_info.getClassName()))
            ww = weka_object_to_oscail(JClass(cn)())
            ww.suffix = suffix
            weka_wrappers.append(ww)
        else:
            print 'ignored'
    dest = op.join(op.dirname(__file__), 'weka_filters_wrappers.py')
    if op.exists(dest):
        os.remove(dest)
    with open(dest, 'w') as writer:
        writer.write('from jpype import *\n')
        writer.write('from oscail.common.integration.jpype_utils import jpype_bootup\n\n')
        writer.write('jpype_bootup()\n\n')
        writer.write('\n'.join(weka_imports) + '\n\n')
        writer.write('\n\n'.join(ww.instantiation_function('WF') for ww in weka_wrappers))


def generate_as_wrappers():
    available_evaluators = discover_subclasses(JClass('weka.attributeSelection.ASEvaluation'))
    available_searchers = discover_subclasses(JClass('weka.attributeSelection.ASSearch'))

    def is_banned(wfilter):
        banned = {
            'weka.classifiers.rules.DecisionTable$DummySubsetEvaluator',
            u'weka.classifiers.rules.DTNB$EvalWithDelete',
            u'weka.classifiers.rules.DTNB$BackwardsWithDelete',
            u'weka.attributeSelection.KMedoidsSampling',
        }
        return wfilter in banned
    weka_wrappers = []
    weka_imports = []
    for i, class_info in enumerate(available_evaluators):
        cn = class_info.getClassName()
        if not is_banned(cn):
            print 'Class %d of %d: %s' % (i, available_evaluators.size(), cn)
        else:
            print 'ignored'
            continue
        weka_imports.append('%s=JClass(\'%s\')' % (cn.rpartition('.')[2], cn))
        ww = weka_object_to_oscail(JClass(cn)())
        weka_wrappers.append(ww)
    for i, class_info in enumerate(available_searchers):
        print 'Class %d of %d: %s' % (i, available_searchers.size(), class_info.getClassName())
        if not is_banned(class_info.getClassName()):
            weka_imports.append('%s=JClass(\'%s\')' % (class_info.getClassName().rpartition('.')[2],
                                                       class_info.getClassName()))
            ww = weka_object_to_oscail(JClass(class_info.getClassName())())
            weka_wrappers.append(ww)
        else:
            print 'ignored'
    dest = op.join(op.dirname(__file__), 'weka_as_wrappers.py')
    if op.exists(dest):
        os.remove(dest)
    with open(dest, 'w') as writer:
        writer.write('from jpype import *\n')
        writer.write('from oscail.common.integration.jpype_utils import jpype_bootup\n\n')
        writer.write('jpype_bootup()\n\n')
        writer.write('\n'.join(weka_imports) + '\n\n')
        writer.write('\n\n'.join(ww.instantiation_function('WAS') for ww in weka_wrappers))


def generate_wrappers():
    jpype_bootup()
    generate_filter_wrappers()      # Filters
    generate_classifier_wrappers()  # Classifiers
    generate_as_wrappers()          # Attribute selection

if __name__ == '__main__':
    generate_wrappers()
