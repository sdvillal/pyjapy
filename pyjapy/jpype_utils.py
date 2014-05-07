# coding=utf-8
"""Some utilities to work with JPype.
  JPype Gotchas:
    - does not support well generics
    - threaded code might fail
    - JArray class is quite rotten (report obvious bugs, there is a github guy fixing some...)
TODO: replace JPype by Pyjnius, as the latter is actively developed.
"""
from glob import glob
import os.path as op
from jpype import *
from jpype._jarray import _getClassFor  # Needed to do reflective method call invocation


def jpype_bootup(heap='4G'):
    # TODO: configurable via property
    # TODO: default java libs dir should be in the source dir, ignored in git (integration/3rdparty)
    JAVA_DIR = op.realpath(op.join(op.dirname(__file__), '..', '..', '..', '..', '..', '..', 'libs', 'java'))
    java_classpath = ':'.join(glob(op.join(JAVA_DIR, '*.jar')))
    options = [
        '-server',
        '-Xmx%s' % heap,
        '-Djava.class.path=%s' % java_classpath]
    if not isJVMStarted():
        startJVM(getDefaultJVMPath(), *options)
        java.lang.System.out.println('I print this from the JVM')


#################################
# Utilities
#################################

def java_class(java_object):
    return java_object.getClass().__javaclass__


def discover_subclasses(java_class):
    """Return a list of the not abstract subclasses of the specified java class."""
    #See http://software.clapper.org/javautil/api/org/clapper/util/classutil/ClassFinder.html
    ClassUtil = JPackage('org.clapper.util.classutil')
    class_finder = ClassUtil.ClassFinder()
    class_finder.addClassPath()
    class_filter = ClassUtil.AndClassFilter()
    class_filter.addFilter(ClassUtil.SubclassClassFilter(java_class))                       # Subclass
    class_filter.addFilter(ClassUtil.NotClassFilter(ClassUtil.AbstractClassFilter()))       # Not abstract
    class_filter.addFilter(ClassUtil.NotClassFilter(ClassUtil.InterfaceOnlyClassFilter()))  # Not interface
    found_classes = JClass('java.util.ArrayList')()
    class_finder.findClasses(found_classes, class_filter)
    return found_classes


def get_class_for(class_st='java.lang.Object[]'):
    return _getClassFor(class_st)

#TODO: report the many bugs in _jwrapper and _jarray, JObject should have a typeName class member