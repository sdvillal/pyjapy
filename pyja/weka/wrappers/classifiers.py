from jpype import *
from oscail.common.integration.jpype_utils import jpype_bootup

jpype_bootup()

GridSearch=JClass('weka.classifiers.meta.GridSearch')
MetaCost=JClass('weka.classifiers.meta.MetaCost')
END=JClass('weka.classifiers.meta.END')
RotationForest=JClass('weka.classifiers.meta.RotationForest')
AdditiveRegression=JClass('weka.classifiers.meta.AdditiveRegression')
RegressionByDiscretization=JClass('weka.classifiers.meta.RegressionByDiscretization')
Bagging=JClass('weka.classifiers.meta.Bagging')
LogitBoost=JClass('weka.classifiers.meta.LogitBoost')
MultiClassClassifierUpdateable=JClass('weka.classifiers.meta.MultiClassClassifierUpdateable')
CVParameterSelection=JClass('weka.classifiers.meta.CVParameterSelection')
RandomSubSpace=JClass('weka.classifiers.meta.RandomSubSpace')
FilteredClassifier=JClass('weka.classifiers.meta.FilteredClassifier')
OrdinalClassClassifier=JClass('weka.classifiers.meta.OrdinalClassClassifier')
Stacking=JClass('weka.classifiers.meta.Stacking')
ThresholdSelector=JClass('weka.classifiers.meta.ThresholdSelector')
MultiBoostAB=JClass('weka.classifiers.meta.MultiBoostAB')
RacedIncrementalLogitBoost=JClass('weka.classifiers.meta.RacedIncrementalLogitBoost')
AttributeSelectedClassifier=JClass('weka.classifiers.meta.AttributeSelectedClassifier')
ClassBalancedND=JClass('weka.classifiers.meta.nestedDichotomies.ClassBalancedND')
ND=JClass('weka.classifiers.meta.nestedDichotomies.ND')
DataNearBalancedND=JClass('weka.classifiers.meta.nestedDichotomies.DataNearBalancedND')
MultiScheme=JClass('weka.classifiers.meta.MultiScheme')
Dagging=JClass('weka.classifiers.meta.Dagging')
Vote=JClass('weka.classifiers.meta.Vote')
StackingC=JClass('weka.classifiers.meta.StackingC')
RandomCommittee=JClass('weka.classifiers.meta.RandomCommittee')
CostSensitiveClassifier=JClass('weka.classifiers.meta.CostSensitiveClassifier')
Decorate=JClass('weka.classifiers.meta.Decorate')
RealAdaBoost=JClass('weka.classifiers.meta.RealAdaBoost')
AdaBoostM1=JClass('weka.classifiers.meta.AdaBoostM1')
MultiClassClassifier=JClass('weka.classifiers.meta.MultiClassClassifier')
ClassificationViaRegression=JClass('weka.classifiers.meta.ClassificationViaRegression')
ClassificationViaClustering=JClass('weka.classifiers.meta.ClassificationViaClustering')
Grading=JClass('weka.classifiers.meta.Grading')
OneClassClassifier=JClass('weka.classifiers.meta.OneClassClassifier')
BFTree=JClass('weka.classifiers.trees.BFTree')
J48=JClass('weka.classifiers.trees.J48')
SimpleCart=JClass('weka.classifiers.trees.SimpleCart')
J48graft=JClass('weka.classifiers.trees.J48graft')
ADTree=JClass('weka.classifiers.trees.ADTree')
FT=JClass('weka.classifiers.trees.FT')
LogisticBase=JClass('weka.classifiers.trees.lmt.LogisticBase')
RandomTree=JClass('weka.classifiers.trees.RandomTree')
NBTree=JClass('weka.classifiers.trees.NBTree')
RandomForest=JClass('weka.classifiers.trees.RandomForest')
M5P=JClass('weka.classifiers.trees.M5P')
LMT=JClass('weka.classifiers.trees.LMT')
REPTree=JClass('weka.classifiers.trees.REPTree')
LADTree=JClass('weka.classifiers.trees.LADTree')
DecisionStump=JClass('weka.classifiers.trees.DecisionStump')
Id3=JClass('weka.classifiers.trees.Id3')
SimpleLinearRegression=JClass('weka.classifiers.functions.SimpleLinearRegression')
GaussianProcesses=JClass('weka.classifiers.functions.GaussianProcesses')
SMO=JClass('weka.classifiers.functions.SMO')
SPegasos=JClass('weka.classifiers.functions.SPegasos')
LibSVM=JClass('weka.classifiers.functions.LibSVM')
Winnow=JClass('weka.classifiers.functions.Winnow')
SimpleLogistic=JClass('weka.classifiers.functions.SimpleLogistic')
SMOreg=JClass('weka.classifiers.functions.SMOreg')
SGD=JClass('weka.classifiers.functions.SGD')
MultilayerPerceptron=JClass('weka.classifiers.functions.MultilayerPerceptron')
LeastMedSq=JClass('weka.classifiers.functions.LeastMedSq')
Logistic=JClass('weka.classifiers.functions.Logistic')
LibLINEAR=JClass('weka.classifiers.functions.LibLINEAR')
RBFRegressor=JClass('weka.classifiers.functions.RBFRegressor')
PLSClassifier=JClass('weka.classifiers.functions.PLSClassifier')
VotedPerceptron=JClass('weka.classifiers.functions.VotedPerceptron')
IsotonicRegression=JClass('weka.classifiers.functions.IsotonicRegression')
LinearRegression=JClass('weka.classifiers.functions.LinearRegression')
RBFNetwork=JClass('weka.classifiers.functions.RBFNetwork')
PaceRegression=JClass('weka.classifiers.functions.PaceRegression')
MultilayerPerceptronCS=JClass('weka.classifiers.functions.MultilayerPerceptronCS')
ComplementNaiveBayes=JClass('weka.classifiers.bayes.ComplementNaiveBayes')
NaiveBayesSimple=JClass('weka.classifiers.bayes.NaiveBayesSimple')
HNB=JClass('weka.classifiers.bayes.HNB')
AODEsr=JClass('weka.classifiers.bayes.AODEsr')
NaiveBayesUpdateable=JClass('weka.classifiers.bayes.NaiveBayesUpdateable')
WAODE=JClass('weka.classifiers.bayes.WAODE')
NaiveBayes=JClass('weka.classifiers.bayes.NaiveBayes')
AODE=JClass('weka.classifiers.bayes.AODE')
NaiveBayesMultinomial=JClass('weka.classifiers.bayes.NaiveBayesMultinomial')
EditableBayesNet=JClass('weka.classifiers.bayes.net.EditableBayesNet')
BayesNetGenerator=JClass('weka.classifiers.bayes.net.BayesNetGenerator')
BIFReader=JClass('weka.classifiers.bayes.net.BIFReader')
BayesNet=JClass('weka.classifiers.bayes.BayesNet')
NaiveBayesMultinomialUpdateable=JClass('weka.classifiers.bayes.NaiveBayesMultinomialUpdateable')
DecisionTable=JClass('weka.classifiers.rules.DecisionTable')
FURIA=JClass('weka.classifiers.rules.FURIA')
ConjunctiveRule=JClass('weka.classifiers.rules.ConjunctiveRule')
JRip=JClass('weka.classifiers.rules.JRip')
Ridor=JClass('weka.classifiers.rules.Ridor')
OLM=JClass('weka.classifiers.rules.OLM')
PART=JClass('weka.classifiers.rules.PART')
M5Rules=JClass('weka.classifiers.rules.M5Rules')
NNge=JClass('weka.classifiers.rules.NNge')
Prism=JClass('weka.classifiers.rules.Prism')
OneR=JClass('weka.classifiers.rules.OneR')
ZeroR=JClass('weka.classifiers.rules.ZeroR')
GroovyClassifier=JClass('weka.classifiers.scripting.GroovyClassifier')
JythonClassifier=JClass('weka.classifiers.scripting.JythonClassifier')
CHIRP=JClass('weka.classifiers.misc.CHIRP')
InputMappedClassifier=JClass('weka.classifiers.misc.InputMappedClassifier')
SerializedClassifier=JClass('weka.classifiers.misc.SerializedClassifier')
VFI=JClass('weka.classifiers.misc.VFI')
HyperPipes=JClass('weka.classifiers.misc.HyperPipes')
OSDL=JClass('weka.classifiers.misc.OSDL')
FLR=JClass('weka.classifiers.misc.FLR')
IB1=JClass('weka.classifiers.lazy.IB1')
LWL=JClass('weka.classifiers.lazy.LWL')
KStar=JClass('weka.classifiers.lazy.KStar')
IBk=JClass('weka.classifiers.lazy.IBk')
LBR=JClass('weka.classifiers.lazy.LBR')

def WCGridSearch(XBase=10.0,
                 XExpression='I',
                 XMax=20.0,
                 XMin=5.0,
                 XProperty='filter.numComponents',
                 XStep=1.0,
                 YBase=10.0,
                 YExpression='pow(BASE,I)',
                 YMax=5.0,
                 YMin=-10.0,
                 YProperty='classifier.ridge',
                 YStep=1.0,
                 classifier=None,
                 debug=False,
                 evaluation=None,
                 filter=None,
                 gridIsExtendable=False,
                 logFile=None,
                 maxGridExtensions=3,
                 numExecutionSlots=1,
                 sampleSizePercent=100.0,
                 seed=1,
                 traversal=None):
    gridsearch = GridSearch()
    if not XBase is None: gridsearch.setXBase(XBase)
    if not XExpression is None: gridsearch.setXExpression(XExpression)
    if not XMax is None: gridsearch.setXMax(XMax)
    if not XMin is None: gridsearch.setXMin(XMin)
    if not XProperty is None: gridsearch.setXProperty(XProperty)
    if not XStep is None: gridsearch.setXStep(XStep)
    if not YBase is None: gridsearch.setYBase(YBase)
    if not YExpression is None: gridsearch.setYExpression(YExpression)
    if not YMax is None: gridsearch.setYMax(YMax)
    if not YMin is None: gridsearch.setYMin(YMin)
    if not YProperty is None: gridsearch.setYProperty(YProperty)
    if not YStep is None: gridsearch.setYStep(YStep)
    if not classifier is None: gridsearch.setClassifier(classifier)
    if not debug is None: gridsearch.setDebug(debug)
    if not evaluation is None: gridsearch.setEvaluation(evaluation)
    if not filter is None: gridsearch.setFilter(filter)
    if not gridIsExtendable is None: gridsearch.setGridIsExtendable(gridIsExtendable)
    if not logFile is None: gridsearch.setLogFile(logFile)
    if not maxGridExtensions is None: gridsearch.setMaxGridExtensions(maxGridExtensions)
    if not numExecutionSlots is None: gridsearch.setNumExecutionSlots(numExecutionSlots)
    if not sampleSizePercent is None: gridsearch.setSampleSizePercent(sampleSizePercent)
    if not seed is None: gridsearch.setSeed(seed)
    if not traversal is None: gridsearch.setTraversal(traversal)
    return gridsearch

def WCMetaCost(bagSizePercent=100,
               classifier=None,
               costMatrix= 0
,
               costMatrixSource=None,
               debug=False,
               numIterations=10,
               onDemandDirectory=None,
               seed=1):
    metacost = MetaCost()
    if not bagSizePercent is None: metacost.setBagSizePercent(bagSizePercent)
    if not classifier is None: metacost.setClassifier(classifier)
    if not costMatrix is None: metacost.setCostMatrix(costMatrix)
    if not costMatrixSource is None: metacost.setCostMatrixSource(costMatrixSource)
    if not debug is None: metacost.setDebug(debug)
    if not numIterations is None: metacost.setNumIterations(numIterations)
    if not onDemandDirectory is None: metacost.setOnDemandDirectory(onDemandDirectory)
    if not seed is None: metacost.setSeed(seed)
    return metacost

def WCEND(classifier=None,
          debug=False,
          numIterations=10,
          seed=1):
    end = END()
    if not classifier is None: end.setClassifier(classifier)
    if not debug is None: end.setDebug(debug)
    if not numIterations is None: end.setNumIterations(numIterations)
    if not seed is None: end.setSeed(seed)
    return end

def WCRotationForest(classifier=None,
                     debug=False,
                     maxGroup=3,
                     minGroup=3,
                     numExecutionSlots=1,
                     numIterations=10,
                     numberOfGroups=False,
                     projectionFilter=None,
                     removedPercentage=50,
                     seed=1):
    rotationforest = RotationForest()
    if not classifier is None: rotationforest.setClassifier(classifier)
    if not debug is None: rotationforest.setDebug(debug)
    if not maxGroup is None: rotationforest.setMaxGroup(maxGroup)
    if not minGroup is None: rotationforest.setMinGroup(minGroup)
    if not numExecutionSlots is None: rotationforest.setNumExecutionSlots(numExecutionSlots)
    if not numIterations is None: rotationforest.setNumIterations(numIterations)
    if not numberOfGroups is None: rotationforest.setNumberOfGroups(numberOfGroups)
    if not projectionFilter is None: rotationforest.setProjectionFilter(projectionFilter)
    if not removedPercentage is None: rotationforest.setRemovedPercentage(removedPercentage)
    if not seed is None: rotationforest.setSeed(seed)
    return rotationforest

def WCAdditiveRegression(classifier=None,
                         debug=False,
                         numIterations=10,
                         shrinkage=1.0):
    additiveregression = AdditiveRegression()
    if not classifier is None: additiveregression.setClassifier(classifier)
    if not debug is None: additiveregression.setDebug(debug)
    if not numIterations is None: additiveregression.setNumIterations(numIterations)
    if not shrinkage is None: additiveregression.setShrinkage(shrinkage)
    return additiveregression

def WCRegressionByDiscretization(classifier=None,
                                 debug=False,
                                 deleteEmptyBins=False,
                                 estimatorType=None,
                                 minimizeAbsoluteError=False,
                                 numBins=10,
                                 useEqualFrequency=False):
    regressionbydiscretization = RegressionByDiscretization()
    if not classifier is None: regressionbydiscretization.setClassifier(classifier)
    if not debug is None: regressionbydiscretization.setDebug(debug)
    if not deleteEmptyBins is None: regressionbydiscretization.setDeleteEmptyBins(deleteEmptyBins)
    if not estimatorType is None: regressionbydiscretization.setEstimatorType(estimatorType)
    if not minimizeAbsoluteError is None: regressionbydiscretization.setMinimizeAbsoluteError(minimizeAbsoluteError)
    if not numBins is None: regressionbydiscretization.setNumBins(numBins)
    if not useEqualFrequency is None: regressionbydiscretization.setUseEqualFrequency(useEqualFrequency)
    return regressionbydiscretization

def WCBagging(bagSizePercent=100,
              calcOutOfBag=False,
              classifier=None,
              debug=False,
              numExecutionSlots=1,
              numIterations=10,
              seed=1):
    bagging = Bagging()
    if not bagSizePercent is None: bagging.setBagSizePercent(bagSizePercent)
    if not calcOutOfBag is None: bagging.setCalcOutOfBag(calcOutOfBag)
    if not classifier is None: bagging.setClassifier(classifier)
    if not debug is None: bagging.setDebug(debug)
    if not numExecutionSlots is None: bagging.setNumExecutionSlots(numExecutionSlots)
    if not numIterations is None: bagging.setNumIterations(numIterations)
    if not seed is None: bagging.setSeed(seed)
    return bagging

def WCLogitBoost(classifier=None,
                 debug=False,
                 likelihoodThreshold=-1.79769313486e+308,
                 numFolds=0,
                 numIterations=10,
                 numRuns=1,
                 seed=1,
                 shrinkage=1.0,
                 useResampling=False,
                 weightThreshold=100):
    logitboost = LogitBoost()
    if not classifier is None: logitboost.setClassifier(classifier)
    if not debug is None: logitboost.setDebug(debug)
    if not likelihoodThreshold is None: logitboost.setLikelihoodThreshold(likelihoodThreshold)
    if not numFolds is None: logitboost.setNumFolds(numFolds)
    if not numIterations is None: logitboost.setNumIterations(numIterations)
    if not numRuns is None: logitboost.setNumRuns(numRuns)
    if not seed is None: logitboost.setSeed(seed)
    if not shrinkage is None: logitboost.setShrinkage(shrinkage)
    if not useResampling is None: logitboost.setUseResampling(useResampling)
    if not weightThreshold is None: logitboost.setWeightThreshold(weightThreshold)
    return logitboost

def WCMultiClassClassifierUpdateable(classifier=None,
                                     debug=False,
                                     method=None,
                                     randomWidthFactor=2.0,
                                     seed=1,
                                     usePairwiseCoupling=False):
    multiclassclassifierupdateable = MultiClassClassifierUpdateable()
    if not classifier is None: multiclassclassifierupdateable.setClassifier(classifier)
    if not debug is None: multiclassclassifierupdateable.setDebug(debug)
    if not method is None: multiclassclassifierupdateable.setMethod(method)
    if not randomWidthFactor is None: multiclassclassifierupdateable.setRandomWidthFactor(randomWidthFactor)
    if not seed is None: multiclassclassifierupdateable.setSeed(seed)
    if not usePairwiseCoupling is None: multiclassclassifierupdateable.setUsePairwiseCoupling(usePairwiseCoupling)
    return multiclassclassifierupdateable

def WCCVParameterSelection(CVParameters=None,
                           classifier=None,
                           debug=False,
                           numFolds=10,
                           seed=1):
    cvparameterselection = CVParameterSelection()
    if not CVParameters is None: cvparameterselection.setCVParameters(CVParameters)
    if not classifier is None: cvparameterselection.setClassifier(classifier)
    if not debug is None: cvparameterselection.setDebug(debug)
    if not numFolds is None: cvparameterselection.setNumFolds(numFolds)
    if not seed is None: cvparameterselection.setSeed(seed)
    return cvparameterselection

def WCRandomSubSpace(classifier=None,
                     debug=False,
                     numExecutionSlots=1,
                     numIterations=10,
                     seed=1,
                     subSpaceSize=0.5):
    randomsubspace = RandomSubSpace()
    if not classifier is None: randomsubspace.setClassifier(classifier)
    if not debug is None: randomsubspace.setDebug(debug)
    if not numExecutionSlots is None: randomsubspace.setNumExecutionSlots(numExecutionSlots)
    if not numIterations is None: randomsubspace.setNumIterations(numIterations)
    if not seed is None: randomsubspace.setSeed(seed)
    if not subSpaceSize is None: randomsubspace.setSubSpaceSize(subSpaceSize)
    return randomsubspace

def WCFilteredClassifier(classifier=None,
                         debug=False,
                         filter=None):
    filteredclassifier = FilteredClassifier()
    if not classifier is None: filteredclassifier.setClassifier(classifier)
    if not debug is None: filteredclassifier.setDebug(debug)
    if not filter is None: filteredclassifier.setFilter(filter)
    return filteredclassifier

def WCOrdinalClassClassifier(classifier=None,
                             debug=False,
                             useSmoothing=True):
    ordinalclassclassifier = OrdinalClassClassifier()
    if not classifier is None: ordinalclassclassifier.setClassifier(classifier)
    if not debug is None: ordinalclassclassifier.setDebug(debug)
    if not useSmoothing is None: ordinalclassclassifier.setUseSmoothing(useSmoothing)
    return ordinalclassclassifier

def WCStacking(classifiers=None,
               debug=False,
               metaClassifier=None,
               numExecutionSlots=1,
               numFolds=10,
               seed=1):
    stacking = Stacking()
    if not classifiers is None: stacking.setClassifiers(classifiers)
    if not debug is None: stacking.setDebug(debug)
    if not metaClassifier is None: stacking.setMetaClassifier(metaClassifier)
    if not numExecutionSlots is None: stacking.setNumExecutionSlots(numExecutionSlots)
    if not numFolds is None: stacking.setNumFolds(numFolds)
    if not seed is None: stacking.setSeed(seed)
    return stacking

def WCThresholdSelector(classifier=None,
                        debug=False,
                        designatedClass=None,
                        evaluationMode=None,
                        manualThresholdValue=-1.0,
                        measure=None,
                        numXValFolds=3,
                        rangeCorrection=None,
                        seed=1):
    thresholdselector = ThresholdSelector()
    if not classifier is None: thresholdselector.setClassifier(classifier)
    if not debug is None: thresholdselector.setDebug(debug)
    if not designatedClass is None: thresholdselector.setDesignatedClass(designatedClass)
    if not evaluationMode is None: thresholdselector.setEvaluationMode(evaluationMode)
    if not manualThresholdValue is None: thresholdselector.setManualThresholdValue(manualThresholdValue)
    if not measure is None: thresholdselector.setMeasure(measure)
    if not numXValFolds is None: thresholdselector.setNumXValFolds(numXValFolds)
    if not rangeCorrection is None: thresholdselector.setRangeCorrection(rangeCorrection)
    if not seed is None: thresholdselector.setSeed(seed)
    return thresholdselector

def WCMultiBoostAB(classifier=None,
                   debug=False,
                   numIterations=10,
                   numSubCmtys=3,
                   seed=1,
                   useResampling=False,
                   weightThreshold=100):
    multiboostab = MultiBoostAB()
    if not classifier is None: multiboostab.setClassifier(classifier)
    if not debug is None: multiboostab.setDebug(debug)
    if not numIterations is None: multiboostab.setNumIterations(numIterations)
    if not numSubCmtys is None: multiboostab.setNumSubCmtys(numSubCmtys)
    if not seed is None: multiboostab.setSeed(seed)
    if not useResampling is None: multiboostab.setUseResampling(useResampling)
    if not weightThreshold is None: multiboostab.setWeightThreshold(weightThreshold)
    return multiboostab

def WCRacedIncrementalLogitBoost(classifier=None,
                                 debug=False,
                                 maxChunkSize=2000,
                                 minChunkSize=500,
                                 pruningType=None,
                                 seed=1,
                                 useResampling=False,
                                 validationChunkSize=1000):
    racedincrementallogitboost = RacedIncrementalLogitBoost()
    if not classifier is None: racedincrementallogitboost.setClassifier(classifier)
    if not debug is None: racedincrementallogitboost.setDebug(debug)
    if not maxChunkSize is None: racedincrementallogitboost.setMaxChunkSize(maxChunkSize)
    if not minChunkSize is None: racedincrementallogitboost.setMinChunkSize(minChunkSize)
    if not pruningType is None: racedincrementallogitboost.setPruningType(pruningType)
    if not seed is None: racedincrementallogitboost.setSeed(seed)
    if not useResampling is None: racedincrementallogitboost.setUseResampling(useResampling)
    if not validationChunkSize is None: racedincrementallogitboost.setValidationChunkSize(validationChunkSize)
    return racedincrementallogitboost

def WCAttributeSelectedClassifier(classifier=None,
                                  debug=False,
                                  evaluator=None,
                                  search=None):
    attributeselectedclassifier = AttributeSelectedClassifier()
    if not classifier is None: attributeselectedclassifier.setClassifier(classifier)
    if not debug is None: attributeselectedclassifier.setDebug(debug)
    if not evaluator is None: attributeselectedclassifier.setEvaluator(evaluator)
    if not search is None: attributeselectedclassifier.setSearch(search)
    return attributeselectedclassifier

def WCClassBalancedND(classifier=None,
                      debug=False,
                      seed=1):
    classbalancednd = ClassBalancedND()
    if not classifier is None: classbalancednd.setClassifier(classifier)
    if not debug is None: classbalancednd.setDebug(debug)
    if not seed is None: classbalancednd.setSeed(seed)
    return classbalancednd

def WCND(classifier=None,
         debug=False,
         seed=1):
    nd = ND()
    if not classifier is None: nd.setClassifier(classifier)
    if not debug is None: nd.setDebug(debug)
    if not seed is None: nd.setSeed(seed)
    return nd

def WCDataNearBalancedND(classifier=None,
                         debug=False,
                         seed=1):
    datanearbalancednd = DataNearBalancedND()
    if not classifier is None: datanearbalancednd.setClassifier(classifier)
    if not debug is None: datanearbalancednd.setDebug(debug)
    if not seed is None: datanearbalancednd.setSeed(seed)
    return datanearbalancednd

def WCMultiScheme(classifiers=None,
                  debug=False,
                  numFolds=0,
                  seed=1):
    multischeme = MultiScheme()
    if not classifiers is None: multischeme.setClassifiers(classifiers)
    if not debug is None: multischeme.setDebug(debug)
    if not numFolds is None: multischeme.setNumFolds(numFolds)
    if not seed is None: multischeme.setSeed(seed)
    return multischeme

def WCDagging(classifier=None,
              debug=False,
              numFolds=10,
              seed=1,
              verbose=False):
    dagging = Dagging()
    if not classifier is None: dagging.setClassifier(classifier)
    if not debug is None: dagging.setDebug(debug)
    if not numFolds is None: dagging.setNumFolds(numFolds)
    if not seed is None: dagging.setSeed(seed)
    if not verbose is None: dagging.setVerbose(verbose)
    return dagging

def WCVote(classifiers=None,
           combinationRule=None,
           debug=False,
           preBuiltClassifiers=None,
           seed=1):
    vote = Vote()
    if not classifiers is None: vote.setClassifiers(classifiers)
    if not combinationRule is None: vote.setCombinationRule(combinationRule)
    if not debug is None: vote.setDebug(debug)
    if not preBuiltClassifiers is None: vote.setPreBuiltClassifiers(preBuiltClassifiers)
    if not seed is None: vote.setSeed(seed)
    return vote

def WCStackingC(classifiers=None,
                debug=False,
                metaClassifier=None,
                numExecutionSlots=1,
                numFolds=10,
                seed=1):
    stackingc = StackingC()
    if not classifiers is None: stackingc.setClassifiers(classifiers)
    if not debug is None: stackingc.setDebug(debug)
    if not metaClassifier is None: stackingc.setMetaClassifier(metaClassifier)
    if not numExecutionSlots is None: stackingc.setNumExecutionSlots(numExecutionSlots)
    if not numFolds is None: stackingc.setNumFolds(numFolds)
    if not seed is None: stackingc.setSeed(seed)
    return stackingc

def WCRandomCommittee(classifier=None,
                      debug=False,
                      numExecutionSlots=1,
                      numIterations=10,
                      seed=1):
    randomcommittee = RandomCommittee()
    if not classifier is None: randomcommittee.setClassifier(classifier)
    if not debug is None: randomcommittee.setDebug(debug)
    if not numExecutionSlots is None: randomcommittee.setNumExecutionSlots(numExecutionSlots)
    if not numIterations is None: randomcommittee.setNumIterations(numIterations)
    if not seed is None: randomcommittee.setSeed(seed)
    return randomcommittee

def WCCostSensitiveClassifier(classifier=None,
                              costMatrix= 0
,
                              costMatrixSource=None,
                              debug=False,
                              minimizeExpectedCost=False,
                              onDemandDirectory=None,
                              seed=1):
    costsensitiveclassifier = CostSensitiveClassifier()
    if not classifier is None: costsensitiveclassifier.setClassifier(classifier)
    if not costMatrix is None: costsensitiveclassifier.setCostMatrix(costMatrix)
    if not costMatrixSource is None: costsensitiveclassifier.setCostMatrixSource(costMatrixSource)
    if not debug is None: costsensitiveclassifier.setDebug(debug)
    if not minimizeExpectedCost is None: costsensitiveclassifier.setMinimizeExpectedCost(minimizeExpectedCost)
    if not onDemandDirectory is None: costsensitiveclassifier.setOnDemandDirectory(onDemandDirectory)
    if not seed is None: costsensitiveclassifier.setSeed(seed)
    return costsensitiveclassifier

def WCDecorate(artificialSize=1.0,
               classifier=None,
               debug=False,
               desiredSize=15,
               numIterations=50,
               seed=1):
    decorate = Decorate()
    if not artificialSize is None: decorate.setArtificialSize(artificialSize)
    if not classifier is None: decorate.setClassifier(classifier)
    if not debug is None: decorate.setDebug(debug)
    if not desiredSize is None: decorate.setDesiredSize(desiredSize)
    if not numIterations is None: decorate.setNumIterations(numIterations)
    if not seed is None: decorate.setSeed(seed)
    return decorate

def WCRealAdaBoost(classifier=None,
                   debug=False,
                   numIterations=10,
                   seed=1,
                   shrinkage=1.0,
                   useResampling=False,
                   weightThreshold=100):
    realadaboost = RealAdaBoost()
    if not classifier is None: realadaboost.setClassifier(classifier)
    if not debug is None: realadaboost.setDebug(debug)
    if not numIterations is None: realadaboost.setNumIterations(numIterations)
    if not seed is None: realadaboost.setSeed(seed)
    if not shrinkage is None: realadaboost.setShrinkage(shrinkage)
    if not useResampling is None: realadaboost.setUseResampling(useResampling)
    if not weightThreshold is None: realadaboost.setWeightThreshold(weightThreshold)
    return realadaboost

def WCAdaBoostM1(classifier=None,
                 debug=False,
                 numIterations=10,
                 seed=1,
                 useResampling=False,
                 weightThreshold=100):
    adaboostm1 = AdaBoostM1()
    if not classifier is None: adaboostm1.setClassifier(classifier)
    if not debug is None: adaboostm1.setDebug(debug)
    if not numIterations is None: adaboostm1.setNumIterations(numIterations)
    if not seed is None: adaboostm1.setSeed(seed)
    if not useResampling is None: adaboostm1.setUseResampling(useResampling)
    if not weightThreshold is None: adaboostm1.setWeightThreshold(weightThreshold)
    return adaboostm1

def WCMultiClassClassifier(classifier=None,
                           debug=False,
                           method=None,
                           randomWidthFactor=2.0,
                           seed=1,
                           usePairwiseCoupling=False):
    multiclassclassifier = MultiClassClassifier()
    if not classifier is None: multiclassclassifier.setClassifier(classifier)
    if not debug is None: multiclassclassifier.setDebug(debug)
    if not method is None: multiclassclassifier.setMethod(method)
    if not randomWidthFactor is None: multiclassclassifier.setRandomWidthFactor(randomWidthFactor)
    if not seed is None: multiclassclassifier.setSeed(seed)
    if not usePairwiseCoupling is None: multiclassclassifier.setUsePairwiseCoupling(usePairwiseCoupling)
    return multiclassclassifier

def WCClassificationViaRegression(classifier=None,
                                  debug=False):
    classificationviaregression = ClassificationViaRegression()
    if not classifier is None: classificationviaregression.setClassifier(classifier)
    if not debug is None: classificationviaregression.setDebug(debug)
    return classificationviaregression

def WCClassificationViaClustering(clusterer=None,
                                  debug=False):
    classificationviaclustering = ClassificationViaClustering()
    if not clusterer is None: classificationviaclustering.setClusterer(clusterer)
    if not debug is None: classificationviaclustering.setDebug(debug)
    return classificationviaclustering

def WCGrading(classifiers=None,
              debug=False,
              metaClassifier=None,
              numExecutionSlots=1,
              numFolds=10,
              seed=1):
    grading = Grading()
    if not classifiers is None: grading.setClassifiers(classifiers)
    if not debug is None: grading.setDebug(debug)
    if not metaClassifier is None: grading.setMetaClassifier(metaClassifier)
    if not numExecutionSlots is None: grading.setNumExecutionSlots(numExecutionSlots)
    if not numFolds is None: grading.setNumFolds(numFolds)
    if not seed is None: grading.setSeed(seed)
    return grading

def WCOneClassClassifier(classifier=None,
                         debug=False,
                         densityOnly=False,
                         nominalGenerator=None,
                         numRepeats=10,
                         numericGenerator=None,
                         percentageHeldout=10.0,
                         proportionGenerated=0.5,
                         seed=1,
                         targetClassLabel='target',
                         targetRejectionRate=0.1,
                         useInstanceWeights=False,
                         useLaplaceCorrection=False):
    oneclassclassifier = OneClassClassifier()
    if not classifier is None: oneclassclassifier.setClassifier(classifier)
    if not debug is None: oneclassclassifier.setDebug(debug)
    if not densityOnly is None: oneclassclassifier.setDensityOnly(densityOnly)
    if not nominalGenerator is None: oneclassclassifier.setNominalGenerator(nominalGenerator)
    if not numRepeats is None: oneclassclassifier.setNumRepeats(numRepeats)
    if not numericGenerator is None: oneclassclassifier.setNumericGenerator(numericGenerator)
    if not percentageHeldout is None: oneclassclassifier.setPercentageHeldout(percentageHeldout)
    if not proportionGenerated is None: oneclassclassifier.setProportionGenerated(proportionGenerated)
    if not seed is None: oneclassclassifier.setSeed(seed)
    if not targetClassLabel is None: oneclassclassifier.setTargetClassLabel(targetClassLabel)
    if not targetRejectionRate is None: oneclassclassifier.setTargetRejectionRate(targetRejectionRate)
    if not useInstanceWeights is None: oneclassclassifier.setUseInstanceWeights(useInstanceWeights)
    if not useLaplaceCorrection is None: oneclassclassifier.setUseLaplaceCorrection(useLaplaceCorrection)
    return oneclassclassifier

def WCBFTree(debug=False,
             heuristic=True,
             minNumObj=2,
             numFoldsPruning=5,
             pruningStrategy=None,
             seed=1,
             sizePer=1.0,
             useErrorRate=True,
             useGini=True,
             useOneSE=False):
    bftree = BFTree()
    if not debug is None: bftree.setDebug(debug)
    if not heuristic is None: bftree.setHeuristic(heuristic)
    if not minNumObj is None: bftree.setMinNumObj(minNumObj)
    if not numFoldsPruning is None: bftree.setNumFoldsPruning(numFoldsPruning)
    if not pruningStrategy is None: bftree.setPruningStrategy(pruningStrategy)
    if not seed is None: bftree.setSeed(seed)
    if not sizePer is None: bftree.setSizePer(sizePer)
    if not useErrorRate is None: bftree.setUseErrorRate(useErrorRate)
    if not useGini is None: bftree.setUseGini(useGini)
    if not useOneSE is None: bftree.setUseOneSE(useOneSE)
    return bftree

def WCJ48(binarySplits=False,
          collapseTree=True,
          confidenceFactor=0.25,
          debug=False,
          minNumObj=2,
          numFolds=3,
          reducedErrorPruning=False,
          saveInstanceData=False,
          seed=1,
          subtreeRaising=True,
          unpruned=False,
          useLaplace=False,
          useMDLcorrection=True):
    j48 = J48()
    if not binarySplits is None: j48.setBinarySplits(binarySplits)
    if not collapseTree is None: j48.setCollapseTree(collapseTree)
    if not confidenceFactor is None: j48.setConfidenceFactor(confidenceFactor)
    if not debug is None: j48.setDebug(debug)
    if not minNumObj is None: j48.setMinNumObj(minNumObj)
    if not numFolds is None: j48.setNumFolds(numFolds)
    if not reducedErrorPruning is None: j48.setReducedErrorPruning(reducedErrorPruning)
    if not saveInstanceData is None: j48.setSaveInstanceData(saveInstanceData)
    if not seed is None: j48.setSeed(seed)
    if not subtreeRaising is None: j48.setSubtreeRaising(subtreeRaising)
    if not unpruned is None: j48.setUnpruned(unpruned)
    if not useLaplace is None: j48.setUseLaplace(useLaplace)
    if not useMDLcorrection is None: j48.setUseMDLcorrection(useMDLcorrection)
    return j48

def WCSimpleCart(debug=False,
                 heuristic=True,
                 minNumObj=2.0,
                 numFoldsPruning=5,
                 seed=1,
                 sizePer=1.0,
                 useOneSE=False,
                 usePrune=True):
    simplecart = SimpleCart()
    if not debug is None: simplecart.setDebug(debug)
    if not heuristic is None: simplecart.setHeuristic(heuristic)
    if not minNumObj is None: simplecart.setMinNumObj(minNumObj)
    if not numFoldsPruning is None: simplecart.setNumFoldsPruning(numFoldsPruning)
    if not seed is None: simplecart.setSeed(seed)
    if not sizePer is None: simplecart.setSizePer(sizePer)
    if not useOneSE is None: simplecart.setUseOneSE(useOneSE)
    if not usePrune is None: simplecart.setUsePrune(usePrune)
    return simplecart

def WCJ48graft(binarySplits=False,
               confidenceFactor=0.25,
               debug=False,
               minNumObj=2,
               relabel=False,
               saveInstanceData=False,
               subtreeRaising=True,
               unpruned=False,
               useLaplace=False):
    j48graft = J48graft()
    if not binarySplits is None: j48graft.setBinarySplits(binarySplits)
    if not confidenceFactor is None: j48graft.setConfidenceFactor(confidenceFactor)
    if not debug is None: j48graft.setDebug(debug)
    if not minNumObj is None: j48graft.setMinNumObj(minNumObj)
    if not relabel is None: j48graft.setRelabel(relabel)
    if not saveInstanceData is None: j48graft.setSaveInstanceData(saveInstanceData)
    if not subtreeRaising is None: j48graft.setSubtreeRaising(subtreeRaising)
    if not unpruned is None: j48graft.setUnpruned(unpruned)
    if not useLaplace is None: j48graft.setUseLaplace(useLaplace)
    return j48graft

def WCADTree(debug=False,
             numOfBoostingIterations=10,
             randomSeed=0,
             saveInstanceData=False,
             searchPath=None):
    adtree = ADTree()
    if not debug is None: adtree.setDebug(debug)
    if not numOfBoostingIterations is None: adtree.setNumOfBoostingIterations(numOfBoostingIterations)
    if not randomSeed is None: adtree.setRandomSeed(randomSeed)
    if not saveInstanceData is None: adtree.setSaveInstanceData(saveInstanceData)
    if not searchPath is None: adtree.setSearchPath(searchPath)
    return adtree

def WCFT(binSplit=False,
         debug=False,
         errorOnProbabilities=False,
         minNumInstances=15,
         modelType=None,
         numBoostingIterations=15,
         useAIC=False,
         weightTrimBeta=0.0):
    ft = FT()
    if not binSplit is None: ft.setBinSplit(binSplit)
    if not debug is None: ft.setDebug(debug)
    if not errorOnProbabilities is None: ft.setErrorOnProbabilities(errorOnProbabilities)
    if not minNumInstances is None: ft.setMinNumInstances(minNumInstances)
    if not modelType is None: ft.setModelType(modelType)
    if not numBoostingIterations is None: ft.setNumBoostingIterations(numBoostingIterations)
    if not useAIC is None: ft.setUseAIC(useAIC)
    if not weightTrimBeta is None: ft.setWeightTrimBeta(weightTrimBeta)
    return ft

def WCLogisticBase(debug=False,
                   maxIterations=500,
                   useAIC=False,
                   weightTrimBeta=0.0):
    logisticbase = LogisticBase()
    if not debug is None: logisticbase.setDebug(debug)
    if not maxIterations is None: logisticbase.setMaxIterations(maxIterations)
    if not useAIC is None: logisticbase.setUseAIC(useAIC)
    if not weightTrimBeta is None: logisticbase.setWeightTrimBeta(weightTrimBeta)
    return logisticbase

def WCRandomTree(KValue=0,
                 allowUnclassifiedInstances=False,
                 debug=False,
                 maxDepth=0,
                 minNum=1.0,
                 numFolds=0,
                 seed=1):
    randomtree = RandomTree()
    if not KValue is None: randomtree.setKValue(KValue)
    if not allowUnclassifiedInstances is None: randomtree.setAllowUnclassifiedInstances(allowUnclassifiedInstances)
    if not debug is None: randomtree.setDebug(debug)
    if not maxDepth is None: randomtree.setMaxDepth(maxDepth)
    if not minNum is None: randomtree.setMinNum(minNum)
    if not numFolds is None: randomtree.setNumFolds(numFolds)
    if not seed is None: randomtree.setSeed(seed)
    return randomtree

def WCNBTree(debug=False):
    nbtree = NBTree()
    if not debug is None: nbtree.setDebug(debug)
    return nbtree

def WCRandomForest(debug=False,
                   maxDepth=0,
                   numExecutionSlots=1,
                   numFeatures=0,
                   numTrees=10,
                   printTrees=False,
                   seed=1):
    randomforest = RandomForest()
    if not debug is None: randomforest.setDebug(debug)
    if not maxDepth is None: randomforest.setMaxDepth(maxDepth)
    if not numExecutionSlots is None: randomforest.setNumExecutionSlots(numExecutionSlots)
    if not numFeatures is None: randomforest.setNumFeatures(numFeatures)
    if not numTrees is None: randomforest.setNumTrees(numTrees)
    if not printTrees is None: randomforest.setPrintTrees(printTrees)
    if not seed is None: randomforest.setSeed(seed)
    return randomforest

def WCM5P(buildRegressionTree=False,
          debug=False,
          minNumInstances=4.0,
          saveInstances=False,
          unpruned=False,
          useUnsmoothed=False):
    m5p = M5P()
    if not buildRegressionTree is None: m5p.setBuildRegressionTree(buildRegressionTree)
    if not debug is None: m5p.setDebug(debug)
    if not minNumInstances is None: m5p.setMinNumInstances(minNumInstances)
    if not saveInstances is None: m5p.setSaveInstances(saveInstances)
    if not unpruned is None: m5p.setUnpruned(unpruned)
    if not useUnsmoothed is None: m5p.setUseUnsmoothed(useUnsmoothed)
    return m5p

def WCLMT(convertNominal=False,
          debug=False,
          errorOnProbabilities=False,
          fastRegression=True,
          minNumInstances=15,
          numBoostingIterations=-1,
          splitOnResiduals=False,
          useAIC=False,
          weightTrimBeta=0.0):
    lmt = LMT()
    if not convertNominal is None: lmt.setConvertNominal(convertNominal)
    if not debug is None: lmt.setDebug(debug)
    if not errorOnProbabilities is None: lmt.setErrorOnProbabilities(errorOnProbabilities)
    if not fastRegression is None: lmt.setFastRegression(fastRegression)
    if not minNumInstances is None: lmt.setMinNumInstances(minNumInstances)
    if not numBoostingIterations is None: lmt.setNumBoostingIterations(numBoostingIterations)
    if not splitOnResiduals is None: lmt.setSplitOnResiduals(splitOnResiduals)
    if not useAIC is None: lmt.setUseAIC(useAIC)
    if not weightTrimBeta is None: lmt.setWeightTrimBeta(weightTrimBeta)
    return lmt

def WCREPTree(debug=False,
              initialCount=0.0,
              maxDepth=-1,
              minNum=2.0,
              minVarianceProp=0.001,
              noPruning=False,
              numFolds=3,
              seed=1,
              spreadInitialCount=False):
    reptree = REPTree()
    if not debug is None: reptree.setDebug(debug)
    if not initialCount is None: reptree.setInitialCount(initialCount)
    if not maxDepth is None: reptree.setMaxDepth(maxDepth)
    if not minNum is None: reptree.setMinNum(minNum)
    if not minVarianceProp is None: reptree.setMinVarianceProp(minVarianceProp)
    if not noPruning is None: reptree.setNoPruning(noPruning)
    if not numFolds is None: reptree.setNumFolds(numFolds)
    if not seed is None: reptree.setSeed(seed)
    if not spreadInitialCount is None: reptree.setSpreadInitialCount(spreadInitialCount)
    return reptree

def WCLADTree(debug=False,
              numOfBoostingIterations=10):
    ladtree = LADTree()
    if not debug is None: ladtree.setDebug(debug)
    if not numOfBoostingIterations is None: ladtree.setNumOfBoostingIterations(numOfBoostingIterations)
    return ladtree

def WCDecisionStump(debug=False):
    decisionstump = DecisionStump()
    if not debug is None: decisionstump.setDebug(debug)
    return decisionstump

def WCId3(debug=False):
    id3 = Id3()
    if not debug is None: id3.setDebug(debug)
    return id3

def WCSimpleLinearRegression(debug=False):
    simplelinearregression = SimpleLinearRegression()
    if not debug is None: simplelinearregression.setDebug(debug)
    return simplelinearregression

def WCGaussianProcesses(debug=False,
                        filterType=None,
                        kernel=None,
                        noise=1.0):
    gaussianprocesses = GaussianProcesses()
    if not debug is None: gaussianprocesses.setDebug(debug)
    if not filterType is None: gaussianprocesses.setFilterType(filterType)
    if not kernel is None: gaussianprocesses.setKernel(kernel)
    if not noise is None: gaussianprocesses.setNoise(noise)
    return gaussianprocesses

def WCSMO(buildLogisticModels=False,
          c=1.0,
          checksTurnedOff=False,
          debug=False,
          epsilon=1e-12,
          filterType=None,
          kernel=None,
          numFolds=-1,
          randomSeed=1,
          toleranceParameter=0.001):
    smo = SMO()
    if not buildLogisticModels is None: smo.setBuildLogisticModels(buildLogisticModels)
    if not c is None: smo.setC(c)
    if not checksTurnedOff is None: smo.setChecksTurnedOff(checksTurnedOff)
    if not debug is None: smo.setDebug(debug)
    if not epsilon is None: smo.setEpsilon(epsilon)
    if not filterType is None: smo.setFilterType(filterType)
    if not kernel is None: smo.setKernel(kernel)
    if not numFolds is None: smo.setNumFolds(numFolds)
    if not randomSeed is None: smo.setRandomSeed(randomSeed)
    if not toleranceParameter is None: smo.setToleranceParameter(toleranceParameter)
    return smo

def WCSPegasos(debug=False,
               dontNormalize=False,
               dontReplaceMissing=False,
               epochs=500,
               lambda_=0.0001,
               lossFunction=None):
    spegasos = SPegasos()
    if not debug is None: spegasos.setDebug(debug)
    if not dontNormalize is None: spegasos.setDontNormalize(dontNormalize)
    if not dontReplaceMissing is None: spegasos.setDontReplaceMissing(dontReplaceMissing)
    if not epochs is None: spegasos.setEpochs(epochs)
    if not lambda_ is None: spegasos.setLambda(lambda_)
    if not lossFunction is None: spegasos.setLossFunction(lossFunction)
    return spegasos

def WCLibSVM(SVMType=None,
             cacheSize=40.0,
             coef0=0.0,
             cost=1.0,
             debug=False,
             degree=3,
             doNotReplaceMissingValues=False,
             eps=0.001,
             gamma=0.0,
             kernelType=None,
             loss=0.1,
             modelFile=None,
             normalize=False,
             nu=0.5,
             probabilityEstimates=False,
             shrinking=True,
             weights=''):
    libsvm = LibSVM()
    if not SVMType is None: libsvm.setSVMType(SVMType)
    if not cacheSize is None: libsvm.setCacheSize(cacheSize)
    if not coef0 is None: libsvm.setCoef0(coef0)
    if not cost is None: libsvm.setCost(cost)
    if not debug is None: libsvm.setDebug(debug)
    if not degree is None: libsvm.setDegree(degree)
    if not doNotReplaceMissingValues is None: libsvm.setDoNotReplaceMissingValues(doNotReplaceMissingValues)
    if not eps is None: libsvm.setEps(eps)
    if not gamma is None: libsvm.setGamma(gamma)
    if not kernelType is None: libsvm.setKernelType(kernelType)
    if not loss is None: libsvm.setLoss(loss)
    if not modelFile is None: libsvm.setModelFile(modelFile)
    if not normalize is None: libsvm.setNormalize(normalize)
    if not nu is None: libsvm.setNu(nu)
    if not probabilityEstimates is None: libsvm.setProbabilityEstimates(probabilityEstimates)
    if not shrinking is None: libsvm.setShrinking(shrinking)
    if not weights is None: libsvm.setWeights(weights)
    return libsvm

def WCWinnow(alpha=2.0,
             balanced=False,
             beta=0.5,
             debug=False,
             defaultWeight=2.0,
             numIterations=1,
             seed=1,
             threshold=-1.0):
    winnow = Winnow()
    if not alpha is None: winnow.setAlpha(alpha)
    if not balanced is None: winnow.setBalanced(balanced)
    if not beta is None: winnow.setBeta(beta)
    if not debug is None: winnow.setDebug(debug)
    if not defaultWeight is None: winnow.setDefaultWeight(defaultWeight)
    if not numIterations is None: winnow.setNumIterations(numIterations)
    if not seed is None: winnow.setSeed(seed)
    if not threshold is None: winnow.setThreshold(threshold)
    return winnow

def WCSimpleLogistic(debug=False,
                     errorOnProbabilities=False,
                     heuristicStop=50,
                     maxBoostingIterations=500,
                     numBoostingIterations=0,
                     useAIC=False,
                     useCrossValidation=True,
                     weightTrimBeta=0.0):
    simplelogistic = SimpleLogistic()
    if not debug is None: simplelogistic.setDebug(debug)
    if not errorOnProbabilities is None: simplelogistic.setErrorOnProbabilities(errorOnProbabilities)
    if not heuristicStop is None: simplelogistic.setHeuristicStop(heuristicStop)
    if not maxBoostingIterations is None: simplelogistic.setMaxBoostingIterations(maxBoostingIterations)
    if not numBoostingIterations is None: simplelogistic.setNumBoostingIterations(numBoostingIterations)
    if not useAIC is None: simplelogistic.setUseAIC(useAIC)
    if not useCrossValidation is None: simplelogistic.setUseCrossValidation(useCrossValidation)
    if not weightTrimBeta is None: simplelogistic.setWeightTrimBeta(weightTrimBeta)
    return simplelogistic

def WCSMOreg(c=1.0,
             debug=False,
             filterType=None,
             kernel=None,
             regOptimizer=None):
    smoreg = SMOreg()
    if not c is None: smoreg.setC(c)
    if not debug is None: smoreg.setDebug(debug)
    if not filterType is None: smoreg.setFilterType(filterType)
    if not kernel is None: smoreg.setKernel(kernel)
    if not regOptimizer is None: smoreg.setRegOptimizer(regOptimizer)
    return smoreg

def WCSGD(debug=False,
          dontNormalize=False,
          dontReplaceMissing=False,
          epochs=500,
          lambda_=0.0001,
          learningRate=0.01,
          lossFunction=None,
          seed=1):
    sgd = SGD()
    if not debug is None: sgd.setDebug(debug)
    if not dontNormalize is None: sgd.setDontNormalize(dontNormalize)
    if not dontReplaceMissing is None: sgd.setDontReplaceMissing(dontReplaceMissing)
    if not epochs is None: sgd.setEpochs(epochs)
    if not lambda_ is None: sgd.setLambda(lambda_)
    if not learningRate is None: sgd.setLearningRate(learningRate)
    if not lossFunction is None: sgd.setLossFunction(lossFunction)
    if not seed is None: sgd.setSeed(seed)
    return sgd

def WCMultilayerPerceptron(GUI=False,
                           autoBuild=True,
                           debug=False,
                           decay=False,
                           hiddenLayers='a',
                           learningRate=0.3,
                           momentum=0.2,
                           nominalToBinaryFilter=True,
                           normalizeAttributes=True,
                           normalizeNumericClass=True,
                           reset=True,
                           seed=0,
                           trainingTime=500,
                           validationSetSize=0,
                           validationThreshold=20):
    multilayerperceptron = MultilayerPerceptron()
    if not GUI is None: multilayerperceptron.setGUI(GUI)
    if not autoBuild is None: multilayerperceptron.setAutoBuild(autoBuild)
    if not debug is None: multilayerperceptron.setDebug(debug)
    if not decay is None: multilayerperceptron.setDecay(decay)
    if not hiddenLayers is None: multilayerperceptron.setHiddenLayers(hiddenLayers)
    if not learningRate is None: multilayerperceptron.setLearningRate(learningRate)
    if not momentum is None: multilayerperceptron.setMomentum(momentum)
    if not nominalToBinaryFilter is None: multilayerperceptron.setNominalToBinaryFilter(nominalToBinaryFilter)
    if not normalizeAttributes is None: multilayerperceptron.setNormalizeAttributes(normalizeAttributes)
    if not normalizeNumericClass is None: multilayerperceptron.setNormalizeNumericClass(normalizeNumericClass)
    if not reset is None: multilayerperceptron.setReset(reset)
    if not seed is None: multilayerperceptron.setSeed(seed)
    if not trainingTime is None: multilayerperceptron.setTrainingTime(trainingTime)
    if not validationSetSize is None: multilayerperceptron.setValidationSetSize(validationSetSize)
    if not validationThreshold is None: multilayerperceptron.setValidationThreshold(validationThreshold)
    return multilayerperceptron

def WCLeastMedSq(debug=False,
                 randomSeed=0,
                 sampleSize=4):
    leastmedsq = LeastMedSq()
    if not debug is None: leastmedsq.setDebug(debug)
    if not randomSeed is None: leastmedsq.setRandomSeed(randomSeed)
    if not sampleSize is None: leastmedsq.setSampleSize(sampleSize)
    return leastmedsq

def WCLogistic(debug=False,
               maxIts=-1,
               ridge=1e-08,
               useConjugateGradientDescent=False):
    logistic = Logistic()
    if not debug is None: logistic.setDebug(debug)
    if not maxIts is None: logistic.setMaxIts(maxIts)
    if not ridge is None: logistic.setRidge(ridge)
    if not useConjugateGradientDescent is None: logistic.setUseConjugateGradientDescent(useConjugateGradientDescent)
    return logistic

def WCLibLINEAR(SVMType=None,
                bias=1.0,
                convertNominalToBinary=False,
                cost=1.0,
                debug=False,
                doNotReplaceMissingValues=False,
                eps=0.01,
                normalize=False,
                probabilityEstimates=False,
                weights=''):
    liblinear = LibLINEAR()
    if not SVMType is None: liblinear.setSVMType(SVMType)
    if not bias is None: liblinear.setBias(bias)
    if not convertNominalToBinary is None: liblinear.setConvertNominalToBinary(convertNominalToBinary)
    if not cost is None: liblinear.setCost(cost)
    if not debug is None: liblinear.setDebug(debug)
    if not doNotReplaceMissingValues is None: liblinear.setDoNotReplaceMissingValues(doNotReplaceMissingValues)
    if not eps is None: liblinear.setEps(eps)
    if not normalize is None: liblinear.setNormalize(normalize)
    if not probabilityEstimates is None: liblinear.setProbabilityEstimates(probabilityEstimates)
    if not weights is None: liblinear.setWeights(weights)
    return liblinear

def WCRBFRegressor(debug=False,
                   numFunctions=2,
                   ridge=0.01,
                   scaleOptimizationOption=None,
                   seed=1,
                   useAttributeWeights=False,
                   useCGD=False,
                   useNormalizedBasisFunctions=False):
    rbfregressor = RBFRegressor()
    if not debug is None: rbfregressor.setDebug(debug)
    if not numFunctions is None: rbfregressor.setNumFunctions(numFunctions)
    if not ridge is None: rbfregressor.setRidge(ridge)
    if not scaleOptimizationOption is None: rbfregressor.setScaleOptimizationOption(scaleOptimizationOption)
    if not seed is None: rbfregressor.setSeed(seed)
    if not useAttributeWeights is None: rbfregressor.setUseAttributeWeights(useAttributeWeights)
    if not useCGD is None: rbfregressor.setUseCGD(useCGD)
    if not useNormalizedBasisFunctions is None: rbfregressor.setUseNormalizedBasisFunctions(useNormalizedBasisFunctions)
    return rbfregressor

def WCPLSClassifier(debug=False,
                    filter=None):
    plsclassifier = PLSClassifier()
    if not debug is None: plsclassifier.setDebug(debug)
    if not filter is None: plsclassifier.setFilter(filter)
    return plsclassifier

def WCVotedPerceptron(debug=False,
                      exponent=1.0,
                      maxK=10000,
                      numIterations=1,
                      seed=1):
    votedperceptron = VotedPerceptron()
    if not debug is None: votedperceptron.setDebug(debug)
    if not exponent is None: votedperceptron.setExponent(exponent)
    if not maxK is None: votedperceptron.setMaxK(maxK)
    if not numIterations is None: votedperceptron.setNumIterations(numIterations)
    if not seed is None: votedperceptron.setSeed(seed)
    return votedperceptron

def WCIsotonicRegression(debug=False):
    isotonicregression = IsotonicRegression()
    if not debug is None: isotonicregression.setDebug(debug)
    return isotonicregression

def WCLinearRegression(attributeSelectionMethod=None,
                       debug=False,
                       eliminateColinearAttributes=True,
                       minimal=False,
                       ridge=1e-08):
    linearregression = LinearRegression()
    if not attributeSelectionMethod is None: linearregression.setAttributeSelectionMethod(attributeSelectionMethod)
    if not debug is None: linearregression.setDebug(debug)
    if not eliminateColinearAttributes is None: linearregression.setEliminateColinearAttributes(eliminateColinearAttributes)
    if not minimal is None: linearregression.setMinimal(minimal)
    if not ridge is None: linearregression.setRidge(ridge)
    return linearregression

def WCRBFNetwork(clusteringSeed=1,
                 debug=False,
                 maxIts=-1,
                 minStdDev=0.1,
                 numClusters=2,
                 ridge=1e-08):
    rbfnetwork = RBFNetwork()
    if not clusteringSeed is None: rbfnetwork.setClusteringSeed(clusteringSeed)
    if not debug is None: rbfnetwork.setDebug(debug)
    if not maxIts is None: rbfnetwork.setMaxIts(maxIts)
    if not minStdDev is None: rbfnetwork.setMinStdDev(minStdDev)
    if not numClusters is None: rbfnetwork.setNumClusters(numClusters)
    if not ridge is None: rbfnetwork.setRidge(ridge)
    return rbfnetwork

def WCPaceRegression(debug=False,
                     estimator=None,
                     threshold=2.0):
    paceregression = PaceRegression()
    if not debug is None: paceregression.setDebug(debug)
    if not estimator is None: paceregression.setEstimator(estimator)
    if not threshold is None: paceregression.setThreshold(threshold)
    return paceregression

def WCMultilayerPerceptronCS(GUI=False,
                             autoBuild=True,
                             debug=False,
                             decay=False,
                             hiddenLayers='a',
                             learningRate=0.3,
                             momentum=0.2,
                             nominalToBinaryFilter=True,
                             normalizeAttributes=True,
                             normalizeNumericClass=True,
                             reset=True,
                             secFile='',
                             seed=0,
                             trainingTime=500,
                             valFile='',
                             validationSetSize=0,
                             validationThreshold=20):
    multilayerperceptroncs = MultilayerPerceptronCS()
    if not GUI is None: multilayerperceptroncs.setGUI(GUI)
    if not autoBuild is None: multilayerperceptroncs.setAutoBuild(autoBuild)
    if not debug is None: multilayerperceptroncs.setDebug(debug)
    if not decay is None: multilayerperceptroncs.setDecay(decay)
    if not hiddenLayers is None: multilayerperceptroncs.setHiddenLayers(hiddenLayers)
    if not learningRate is None: multilayerperceptroncs.setLearningRate(learningRate)
    if not momentum is None: multilayerperceptroncs.setMomentum(momentum)
    if not nominalToBinaryFilter is None: multilayerperceptroncs.setNominalToBinaryFilter(nominalToBinaryFilter)
    if not normalizeAttributes is None: multilayerperceptroncs.setNormalizeAttributes(normalizeAttributes)
    if not normalizeNumericClass is None: multilayerperceptroncs.setNormalizeNumericClass(normalizeNumericClass)
    if not reset is None: multilayerperceptroncs.setReset(reset)
    if not secFile is None: multilayerperceptroncs.setSecFile(secFile)
    if not seed is None: multilayerperceptroncs.setSeed(seed)
    if not trainingTime is None: multilayerperceptroncs.setTrainingTime(trainingTime)
    if not valFile is None: multilayerperceptroncs.setValFile(valFile)
    if not validationSetSize is None: multilayerperceptroncs.setValidationSetSize(validationSetSize)
    if not validationThreshold is None: multilayerperceptroncs.setValidationThreshold(validationThreshold)
    return multilayerperceptroncs

def WCComplementNaiveBayes(debug=False,
                           normalizeWordWeights=False,
                           smoothingParameter=1.0):
    complementnaivebayes = ComplementNaiveBayes()
    if not debug is None: complementnaivebayes.setDebug(debug)
    if not normalizeWordWeights is None: complementnaivebayes.setNormalizeWordWeights(normalizeWordWeights)
    if not smoothingParameter is None: complementnaivebayes.setSmoothingParameter(smoothingParameter)
    return complementnaivebayes

def WCNaiveBayesSimple(debug=False):
    naivebayessimple = NaiveBayesSimple()
    if not debug is None: naivebayessimple.setDebug(debug)
    return naivebayessimple

def WCHNB(debug=False):
    hnb = HNB()
    if not debug is None: hnb.setDebug(debug)
    return hnb

def WCAODEsr(criticalValue=50,
             debug=False,
             frequencyLimit=1,
             mestWeight=1.0,
             useLaplace=False):
    aodesr = AODEsr()
    if not criticalValue is None: aodesr.setCriticalValue(criticalValue)
    if not debug is None: aodesr.setDebug(debug)
    if not frequencyLimit is None: aodesr.setFrequencyLimit(frequencyLimit)
    if not mestWeight is None: aodesr.setMestWeight(mestWeight)
    if not useLaplace is None: aodesr.setUseLaplace(useLaplace)
    return aodesr

def WCNaiveBayesUpdateable(debug=False,
                           displayModelInOldFormat=False,
                           useKernelEstimator=False,
                           useSupervisedDiscretization=False):
    naivebayesupdateable = NaiveBayesUpdateable()
    if not debug is None: naivebayesupdateable.setDebug(debug)
    if not displayModelInOldFormat is None: naivebayesupdateable.setDisplayModelInOldFormat(displayModelInOldFormat)
    if not useKernelEstimator is None: naivebayesupdateable.setUseKernelEstimator(useKernelEstimator)
    if not useSupervisedDiscretization is None: naivebayesupdateable.setUseSupervisedDiscretization(useSupervisedDiscretization)
    return naivebayesupdateable

def WCWAODE(debug=False,
            internals=False):
    waode = WAODE()
    if not debug is None: waode.setDebug(debug)
    if not internals is None: waode.setInternals(internals)
    return waode

def WCNaiveBayes(debug=False,
                 displayModelInOldFormat=False,
                 useKernelEstimator=False,
                 useSupervisedDiscretization=False):
    naivebayes = NaiveBayes()
    if not debug is None: naivebayes.setDebug(debug)
    if not displayModelInOldFormat is None: naivebayes.setDisplayModelInOldFormat(displayModelInOldFormat)
    if not useKernelEstimator is None: naivebayes.setUseKernelEstimator(useKernelEstimator)
    if not useSupervisedDiscretization is None: naivebayes.setUseSupervisedDiscretization(useSupervisedDiscretization)
    return naivebayes

def WCAODE(debug=False,
           frequencyLimit=1,
           useMEstimates=False,
           weight=1):
    aode = AODE()
    if not debug is None: aode.setDebug(debug)
    if not frequencyLimit is None: aode.setFrequencyLimit(frequencyLimit)
    if not useMEstimates is None: aode.setUseMEstimates(useMEstimates)
    if not weight is None: aode.setWeight(weight)
    return aode

def WCNaiveBayesMultinomial(debug=False):
    naivebayesmultinomial = NaiveBayesMultinomial()
    if not debug is None: naivebayesmultinomial.setDebug(debug)
    return naivebayesmultinomial

def WCEditableBayesNet(BIFFile='',
                       debug=False,
                       estimator=None,
                       searchAlgorithm=None,
                       useADTree=False):
    editablebayesnet = EditableBayesNet()
    if not BIFFile is None: editablebayesnet.setBIFFile(BIFFile)
    if not debug is None: editablebayesnet.setDebug(debug)
    if not estimator is None: editablebayesnet.setEstimator(estimator)
    if not searchAlgorithm is None: editablebayesnet.setSearchAlgorithm(searchAlgorithm)
    if not useADTree is None: editablebayesnet.setUseADTree(useADTree)
    return editablebayesnet

def WCBayesNetGenerator(BIFFile='',
                        debug=False,
                        estimator=None,
                        searchAlgorithm=None,
                        useADTree=False):
    bayesnetgenerator = BayesNetGenerator()
    if not BIFFile is None: bayesnetgenerator.setBIFFile(BIFFile)
    if not debug is None: bayesnetgenerator.setDebug(debug)
    if not estimator is None: bayesnetgenerator.setEstimator(estimator)
    if not searchAlgorithm is None: bayesnetgenerator.setSearchAlgorithm(searchAlgorithm)
    if not useADTree is None: bayesnetgenerator.setUseADTree(useADTree)
    return bayesnetgenerator

def WCBIFReader(BIFFile='',
                debug=False,
                estimator=None,
                searchAlgorithm=None,
                useADTree=False):
    bifreader = BIFReader()
    if not BIFFile is None: bifreader.setBIFFile(BIFFile)
    if not debug is None: bifreader.setDebug(debug)
    if not estimator is None: bifreader.setEstimator(estimator)
    if not searchAlgorithm is None: bifreader.setSearchAlgorithm(searchAlgorithm)
    if not useADTree is None: bifreader.setUseADTree(useADTree)
    return bifreader

def WCBayesNet(BIFFile='',
               debug=False,
               estimator=None,
               searchAlgorithm=None,
               useADTree=False):
    bayesnet = BayesNet()
    if not BIFFile is None: bayesnet.setBIFFile(BIFFile)
    if not debug is None: bayesnet.setDebug(debug)
    if not estimator is None: bayesnet.setEstimator(estimator)
    if not searchAlgorithm is None: bayesnet.setSearchAlgorithm(searchAlgorithm)
    if not useADTree is None: bayesnet.setUseADTree(useADTree)
    return bayesnet

def WCNaiveBayesMultinomialUpdateable(debug=False):
    naivebayesmultinomialupdateable = NaiveBayesMultinomialUpdateable()
    if not debug is None: naivebayesmultinomialupdateable.setDebug(debug)
    return naivebayesmultinomialupdateable

def WCDecisionTable(crossVal=1,
                    debug=False,
                    displayRules=False,
                    evaluationMeasure=None,
                    search=None,
                    useIBk=False):
    decisiontable = DecisionTable()
    if not crossVal is None: decisiontable.setCrossVal(crossVal)
    if not debug is None: decisiontable.setDebug(debug)
    if not displayRules is None: decisiontable.setDisplayRules(displayRules)
    if not evaluationMeasure is None: decisiontable.setEvaluationMeasure(evaluationMeasure)
    if not search is None: decisiontable.setSearch(search)
    if not useIBk is None: decisiontable.setUseIBk(useIBk)
    return decisiontable

def WCFURIA(TNorm=None,
            checkErrorRate=True,
            debug=False,
            folds=3,
            minNo=2.0,
            optimizations=2,
            seed=1,
            uncovAction=None):
    furia = FURIA()
    if not TNorm is None: furia.setTNorm(TNorm)
    if not checkErrorRate is None: furia.setCheckErrorRate(checkErrorRate)
    if not debug is None: furia.setDebug(debug)
    if not folds is None: furia.setFolds(folds)
    if not minNo is None: furia.setMinNo(minNo)
    if not optimizations is None: furia.setOptimizations(optimizations)
    if not seed is None: furia.setSeed(seed)
    if not uncovAction is None: furia.setUncovAction(uncovAction)
    return furia

def WCConjunctiveRule(debug=False,
                      exclusive=False,
                      folds=3,
                      minNo=2.0,
                      numAntds=-1,
                      seed=1):
    conjunctiverule = ConjunctiveRule()
    if not debug is None: conjunctiverule.setDebug(debug)
    if not exclusive is None: conjunctiverule.setExclusive(exclusive)
    if not folds is None: conjunctiverule.setFolds(folds)
    if not minNo is None: conjunctiverule.setMinNo(minNo)
    if not numAntds is None: conjunctiverule.setNumAntds(numAntds)
    if not seed is None: conjunctiverule.setSeed(seed)
    return conjunctiverule

def WCJRip(checkErrorRate=True,
           debug=False,
           folds=3,
           minNo=2.0,
           optimizations=2,
           seed=1,
           usePruning=True):
    jrip = JRip()
    if not checkErrorRate is None: jrip.setCheckErrorRate(checkErrorRate)
    if not debug is None: jrip.setDebug(debug)
    if not folds is None: jrip.setFolds(folds)
    if not minNo is None: jrip.setMinNo(minNo)
    if not optimizations is None: jrip.setOptimizations(optimizations)
    if not seed is None: jrip.setSeed(seed)
    if not usePruning is None: jrip.setUsePruning(usePruning)
    return jrip

def WCRidor(debug=False,
            folds=3,
            majorityClass=False,
            minNo=2.0,
            seed=1,
            shuffle=1,
            wholeDataErr=False):
    ridor = Ridor()
    if not debug is None: ridor.setDebug(debug)
    if not folds is None: ridor.setFolds(folds)
    if not majorityClass is None: ridor.setMajorityClass(majorityClass)
    if not minNo is None: ridor.setMinNo(minNo)
    if not seed is None: ridor.setSeed(seed)
    if not shuffle is None: ridor.setShuffle(shuffle)
    if not wholeDataErr is None: ridor.setWholeDataErr(wholeDataErr)
    return ridor

def WCOLM(classificationMode=None,
          debug=False,
          resolutionMode=None,
          ruleSize=-1):
    olm = OLM()
    if not classificationMode is None: olm.setClassificationMode(classificationMode)
    if not debug is None: olm.setDebug(debug)
    if not resolutionMode is None: olm.setResolutionMode(resolutionMode)
    if not ruleSize is None: olm.setRuleSize(ruleSize)
    return olm

def WCPART(binarySplits=False,
           confidenceFactor=0.25,
           debug=False,
           minNumObj=2,
           numFolds=3,
           reducedErrorPruning=False,
           seed=1,
           unpruned=False,
           useMDLcorrection=True):
    part = PART()
    if not binarySplits is None: part.setBinarySplits(binarySplits)
    if not confidenceFactor is None: part.setConfidenceFactor(confidenceFactor)
    if not debug is None: part.setDebug(debug)
    if not minNumObj is None: part.setMinNumObj(minNumObj)
    if not numFolds is None: part.setNumFolds(numFolds)
    if not reducedErrorPruning is None: part.setReducedErrorPruning(reducedErrorPruning)
    if not seed is None: part.setSeed(seed)
    if not unpruned is None: part.setUnpruned(unpruned)
    if not useMDLcorrection is None: part.setUseMDLcorrection(useMDLcorrection)
    return part

def WCM5Rules(buildRegressionTree=False,
              debug=False,
              minNumInstances=4.0,
              unpruned=False,
              useUnsmoothed=False):
    m5rules = M5Rules()
    if not buildRegressionTree is None: m5rules.setBuildRegressionTree(buildRegressionTree)
    if not debug is None: m5rules.setDebug(debug)
    if not minNumInstances is None: m5rules.setMinNumInstances(minNumInstances)
    if not unpruned is None: m5rules.setUnpruned(unpruned)
    if not useUnsmoothed is None: m5rules.setUseUnsmoothed(useUnsmoothed)
    return m5rules

def WCNNge(debug=False,
           numAttemptsOfGeneOption=5,
           numFoldersMIOption=5):
    nnge = NNge()
    if not debug is None: nnge.setDebug(debug)
    if not numAttemptsOfGeneOption is None: nnge.setNumAttemptsOfGeneOption(numAttemptsOfGeneOption)
    if not numFoldersMIOption is None: nnge.setNumFoldersMIOption(numFoldersMIOption)
    return nnge

def WCPrism(debug=False):
    prism = Prism()
    if not debug is None: prism.setDebug(debug)
    return prism

def WCOneR(debug=False,
           minBucketSize=6):
    oner = OneR()
    if not debug is None: oner.setDebug(debug)
    if not minBucketSize is None: oner.setMinBucketSize(minBucketSize)
    return oner

def WCZeroR(debug=False):
    zeror = ZeroR()
    if not debug is None: zeror.setDebug(debug)
    return zeror

def WCGroovyClassifier(debug=False,
                       groovyModule=None,
                       groovyOptions=''):
    groovyclassifier = GroovyClassifier()
    if not debug is None: groovyclassifier.setDebug(debug)
    if not groovyModule is None: groovyclassifier.setGroovyModule(groovyModule)
    if not groovyOptions is None: groovyclassifier.setGroovyOptions(groovyOptions)
    return groovyclassifier

def WCJythonClassifier(debug=False,
                       jythonModule=None,
                       jythonOptions='',
                       jythonPaths=''):
    jythonclassifier = JythonClassifier()
    if not debug is None: jythonclassifier.setDebug(debug)
    if not jythonModule is None: jythonclassifier.setJythonModule(jythonModule)
    if not jythonOptions is None: jythonclassifier.setJythonOptions(jythonOptions)
    if not jythonPaths is None: jythonclassifier.setJythonPaths(jythonPaths)
    return jythonclassifier

def WCCHIRP(debug=False,
            numVoters=7,
            seed=1):
    chirp = CHIRP()
    if not debug is None: chirp.setDebug(debug)
    if not numVoters is None: chirp.setNumVoters(numVoters)
    if not seed is None: chirp.setSeed(seed)
    return chirp

def WCInputMappedClassifier(classifier=None,
                            ignoreCaseForNames=True,
                            modelPath='',
                            suppressMappingReport=False,
                            trim=True):
    inputmappedclassifier = InputMappedClassifier()
    if not classifier is None: inputmappedclassifier.setClassifier(classifier)
    if not ignoreCaseForNames is None: inputmappedclassifier.setIgnoreCaseForNames(ignoreCaseForNames)
    if not modelPath is None: inputmappedclassifier.setModelPath(modelPath)
    if not suppressMappingReport is None: inputmappedclassifier.setSuppressMappingReport(suppressMappingReport)
    if not trim is None: inputmappedclassifier.setTrim(trim)
    return inputmappedclassifier

def WCSerializedClassifier(debug=False,
                           modelFile=None):
    serializedclassifier = SerializedClassifier()
    if not debug is None: serializedclassifier.setDebug(debug)
    if not modelFile is None: serializedclassifier.setModelFile(modelFile)
    return serializedclassifier

def WCVFI(bias=0.6,
          debug=False,
          weightByConfidence=True):
    vfi = VFI()
    if not bias is None: vfi.setBias(bias)
    if not debug is None: vfi.setDebug(debug)
    if not weightByConfidence is None: vfi.setWeightByConfidence(weightByConfidence)
    return vfi

def WCHyperPipes(debug=False):
    hyperpipes = HyperPipes()
    if not debug is None: hyperpipes.setDebug(debug)
    return hyperpipes

def WCOSDL(balanced=False,
           classificationType=None,
           debug=False,
           interpolationParameter=0.5,
           interpolationParameterLowerBound=0.0,
           interpolationParameterUpperBound=1.0,
           numberOfPartsForInterpolationParameter=10,
           tuneInterpolationParameter=False,
           weighted=False):
    osdl = OSDL()
    if not balanced is None: osdl.setBalanced(balanced)
    if not classificationType is None: osdl.setClassificationType(classificationType)
    if not debug is None: osdl.setDebug(debug)
    if not interpolationParameter is None: osdl.setInterpolationParameter(interpolationParameter)
    if not interpolationParameterLowerBound is None: osdl.setInterpolationParameterLowerBound(interpolationParameterLowerBound)
    if not interpolationParameterUpperBound is None: osdl.setInterpolationParameterUpperBound(interpolationParameterUpperBound)
    if not numberOfPartsForInterpolationParameter is None: osdl.setNumberOfPartsForInterpolationParameter(numberOfPartsForInterpolationParameter)
    if not tuneInterpolationParameter is None: osdl.setTuneInterpolationParameter(tuneInterpolationParameter)
    if not weighted is None: osdl.setWeighted(weighted)
    return osdl

def WCFLR(boundsFile='',
          debug=False,
          rhoa=0.5,
          showRules=True):
    flr = FLR()
    if not boundsFile is None: flr.setBoundsFile(boundsFile)
    if not debug is None: flr.setDebug(debug)
    if not rhoa is None: flr.setRhoa(rhoa)
    if not showRules is None: flr.setShowRules(showRules)
    return flr

def WCIB1(debug=False):
    ib1 = IB1()
    if not debug is None: ib1.setDebug(debug)
    return ib1

def WCLWL(KNN=-1,
          classifier=None,
          debug=False,
          nearestNeighbourSearchAlgorithm=None,
          weightingKernel=0):
    lwl = LWL()
    if not KNN is None: lwl.setKNN(KNN)
    if not classifier is None: lwl.setClassifier(classifier)
    if not debug is None: lwl.setDebug(debug)
    if not nearestNeighbourSearchAlgorithm is None: lwl.setNearestNeighbourSearchAlgorithm(nearestNeighbourSearchAlgorithm)
    if not weightingKernel is None: lwl.setWeightingKernel(weightingKernel)
    return lwl

def WCKStar(debug=False,
            entropicAutoBlend=False,
            globalBlend=20,
            missingMode=None):
    kstar = KStar()
    if not debug is None: kstar.setDebug(debug)
    if not entropicAutoBlend is None: kstar.setEntropicAutoBlend(entropicAutoBlend)
    if not globalBlend is None: kstar.setGlobalBlend(globalBlend)
    if not missingMode is None: kstar.setMissingMode(missingMode)
    return kstar

def WCIBk(KNN=1,
          crossValidate=False,
          debug=False,
          distanceWeighting=None,
          meanSquared=False,
          nearestNeighbourSearchAlgorithm=None,
          windowSize=0):
    ibk = IBk()
    if not KNN is None: ibk.setKNN(KNN)
    if not crossValidate is None: ibk.setCrossValidate(crossValidate)
    if not debug is None: ibk.setDebug(debug)
    if not distanceWeighting is None: ibk.setDistanceWeighting(distanceWeighting)
    if not meanSquared is None: ibk.setMeanSquared(meanSquared)
    if not nearestNeighbourSearchAlgorithm is None: ibk.setNearestNeighbourSearchAlgorithm(nearestNeighbourSearchAlgorithm)
    if not windowSize is None: ibk.setWindowSize(windowSize)
    return ibk

def WCLBR(debug=False):
    lbr = LBR()
    if not debug is None: lbr.setDebug(debug)
    return lbr