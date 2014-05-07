from jpype import *
from oscail.common.integration.jpype_utils import jpype_bootup

jpype_bootup()

InvertDataSet=JClass('weka.attributeSelection.ssf.preProcess.InvertDataSet')
MultiFilter=JClass('weka.filters.MultiFilter')
AllFilter=JClass('weka.filters.AllFilter')
StratifiedRemoveFolds_si=JClass('weka.filters.supervised.instance.StratifiedRemoveFolds')
DistributionBasedBalance_si=JClass('weka.filters.supervised.instance.DistributionBasedBalance')
SMOTE_si=JClass('weka.filters.supervised.instance.SMOTE')
Resample_si=JClass('weka.filters.supervised.instance.Resample')
SpreadSubsample_si=JClass('weka.filters.supervised.instance.SpreadSubsample')
Discretize_sa=JClass('weka.filters.supervised.attribute.Discretize')
AddClassification_sa=JClass('weka.filters.supervised.attribute.AddClassification')
NominalToBinary_sa=JClass('weka.filters.supervised.attribute.NominalToBinary')
AttributeSelection_sa=JClass('weka.filters.supervised.attribute.AttributeSelection')
PLSFilter_sa=JClass('weka.filters.supervised.attribute.PLSFilter')
ClassOrder_sa=JClass('weka.filters.supervised.attribute.ClassOrder')
SubsetByExpression_ui=JClass('weka.filters.unsupervised.instance.SubsetByExpression')
NonSparseToSparse_ui=JClass('weka.filters.unsupervised.instance.NonSparseToSparse')
RemovePercentage_ui=JClass('weka.filters.unsupervised.instance.RemovePercentage')
SparseToNonSparse_ui=JClass('weka.filters.unsupervised.instance.SparseToNonSparse')
Denormalize_ui=JClass('weka.filters.unsupervised.instance.Denormalize')
RemoveFolds_ui=JClass('weka.filters.unsupervised.instance.RemoveFolds')
Normalize_ui=JClass('weka.filters.unsupervised.instance.Normalize')
RemoveWithValues_ui=JClass('weka.filters.unsupervised.instance.RemoveWithValues')
Randomize_ui=JClass('weka.filters.unsupervised.instance.Randomize')
RemoveMisclassified_ui=JClass('weka.filters.unsupervised.instance.RemoveMisclassified')
RemoveRange_ui=JClass('weka.filters.unsupervised.instance.RemoveRange')
Resample_ui=JClass('weka.filters.unsupervised.instance.Resample')
RemoveFrequentValues_ui=JClass('weka.filters.unsupervised.instance.RemoveFrequentValues')
ReservoirSample_ui=JClass('weka.filters.unsupervised.instance.ReservoirSample')
Discretize_ua=JClass('weka.filters.unsupervised.attribute.Discretize')
NumericCleaner_ua=JClass('weka.filters.unsupervised.attribute.NumericCleaner')
Wavelet_ua=JClass('weka.filters.unsupervised.attribute.Wavelet')
Center_ua=JClass('weka.filters.unsupervised.attribute.Center')
NominalToString_ua=JClass('weka.filters.unsupervised.attribute.NominalToString')
EMImputation_ua=JClass('weka.filters.unsupervised.attribute.EMImputation')
Standardize_ua=JClass('weka.filters.unsupervised.attribute.Standardize')
TimeSeriesTranslate_ua=JClass('weka.filters.unsupervised.attribute.TimeSeriesTranslate')
TimeSeriesDelta_ua=JClass('weka.filters.unsupervised.attribute.TimeSeriesDelta')
NominalToBinary_ua=JClass('weka.filters.unsupervised.attribute.NominalToBinary')
ReplaceMissingValues_ua=JClass('weka.filters.unsupervised.attribute.ReplaceMissingValues')
PartitionedMultiFilter_ua=JClass('weka.filters.unsupervised.attribute.PartitionedMultiFilter')
AddValues_ua=JClass('weka.filters.unsupervised.attribute.AddValues')
AddCluster_ua=JClass('weka.filters.unsupervised.attribute.AddCluster')
RemoveType_ua=JClass('weka.filters.unsupervised.attribute.RemoveType')
Reorder_ua=JClass('weka.filters.unsupervised.attribute.Reorder')
SortLabels_ua=JClass('weka.filters.unsupervised.attribute.SortLabels')
Normalize_ua=JClass('weka.filters.unsupervised.attribute.Normalize')
MakeIndicator_ua=JClass('weka.filters.unsupervised.attribute.MakeIndicator')
SwapValues_ua=JClass('weka.filters.unsupervised.attribute.SwapValues')
InterquartileRange_ua=JClass('weka.filters.unsupervised.attribute.InterquartileRange')
Remove_ua=JClass('weka.filters.unsupervised.attribute.Remove')
PrincipalComponents_ua=JClass('weka.filters.unsupervised.attribute.PrincipalComponents')
ClassAssigner_ua=JClass('weka.filters.unsupervised.attribute.ClassAssigner')
AddExpression_ua=JClass('weka.filters.unsupervised.attribute.AddExpression')
NumericTransform_ua=JClass('weka.filters.unsupervised.attribute.NumericTransform')
MathExpression_ua=JClass('weka.filters.unsupervised.attribute.MathExpression')
NumericToBinary_ua=JClass('weka.filters.unsupervised.attribute.NumericToBinary')
PropositionalToMultiInstance_ua=JClass('weka.filters.unsupervised.attribute.PropositionalToMultiInstance')
RandomProjection_ua=JClass('weka.filters.unsupervised.attribute.RandomProjection')
AddNoise_ua=JClass('weka.filters.unsupervised.attribute.AddNoise')
FirstOrder_ua=JClass('weka.filters.unsupervised.attribute.FirstOrder')
MILESFilter_ua=JClass('weka.filters.unsupervised.attribute.MILESFilter')
RandomSubset_ua=JClass('weka.filters.unsupervised.attribute.RandomSubset')
KernelFilter_ua=JClass('weka.filters.unsupervised.attribute.KernelFilter')
MergeTwoValues_ua=JClass('weka.filters.unsupervised.attribute.MergeTwoValues')
ClusterMembership_ua=JClass('weka.filters.unsupervised.attribute.ClusterMembership')
AddID_ua=JClass('weka.filters.unsupervised.attribute.AddID')
MultiInstanceToPropositional_ua=JClass('weka.filters.unsupervised.attribute.MultiInstanceToPropositional')
StringToNominal_ua=JClass('weka.filters.unsupervised.attribute.StringToNominal')
RemoveUseless_ua=JClass('weka.filters.unsupervised.attribute.RemoveUseless')
RenameAttribute_ua=JClass('weka.filters.unsupervised.attribute.RenameAttribute')
MergeManyValues_ua=JClass('weka.filters.unsupervised.attribute.MergeManyValues')
Obfuscate_ua=JClass('weka.filters.unsupervised.attribute.Obfuscate')
RELAGGS_ua=JClass('weka.filters.unsupervised.attribute.RELAGGS')
NumericToNominal_ua=JClass('weka.filters.unsupervised.attribute.NumericToNominal')
Copy_ua=JClass('weka.filters.unsupervised.attribute.Copy')
PKIDiscretize_ua=JClass('weka.filters.unsupervised.attribute.PKIDiscretize')
RemoveByName_ua=JClass('weka.filters.unsupervised.attribute.RemoveByName')

def WFInvertDataSet(debug=False):
    invertdataset = InvertDataSet()
    if not debug is None: invertdataset.setDebug(debug)
    return invertdataset

def WFMultiFilter(debug=False,
                  filters=None):
    multifilter = MultiFilter()
    if not debug is None: multifilter.setDebug(debug)
    if not filters is None: multifilter.setFilters(filters)
    return multifilter

def WFAllFilter():
    allfilter = AllFilter()
    return allfilter

def WFStratifiedRemoveFolds_si(fold=1,
                               invertSelection=False,
                               numFolds=10,
                               seed=0):
    stratifiedremovefolds_si = StratifiedRemoveFolds_si()
    if not fold is None: stratifiedremovefolds_si.setFold(fold)
    if not invertSelection is None: stratifiedremovefolds_si.setInvertSelection(invertSelection)
    if not numFolds is None: stratifiedremovefolds_si.setNumFolds(numFolds)
    if not seed is None: stratifiedremovefolds_si.setSeed(seed)
    return stratifiedremovefolds_si

def WFDistributionBasedBalance_si(allowNegativeValues=False,
                                  allowPoissonApproximation=True,
                                  balanceType=None,
                                  invertSelection=False,
                                  labelsRange='first-last',
                                  p=30,
                                  seed=1787586220):
    distributionbasedbalance_si = DistributionBasedBalance_si()
    if not allowNegativeValues is None: distributionbasedbalance_si.setAllowNegativeValues(allowNegativeValues)
    if not allowPoissonApproximation is None: distributionbasedbalance_si.setAllowPoissonApproximation(allowPoissonApproximation)
    if not balanceType is None: distributionbasedbalance_si.setBalanceType(balanceType)
    if not invertSelection is None: distributionbasedbalance_si.setInvertSelection(invertSelection)
    if not labelsRange is None: distributionbasedbalance_si.setLabelsRange(labelsRange)
    if not p is None: distributionbasedbalance_si.setP(p)
    if not seed is None: distributionbasedbalance_si.setSeed(seed)
    return distributionbasedbalance_si

def WFSMOTE_si(classValue='0',
               nearestNeighbors=5,
               percentage=100.0,
               randomSeed=1):
    smote_si = SMOTE_si()
    if not classValue is None: smote_si.setClassValue(classValue)
    if not nearestNeighbors is None: smote_si.setNearestNeighbors(nearestNeighbors)
    if not percentage is None: smote_si.setPercentage(percentage)
    if not randomSeed is None: smote_si.setRandomSeed(randomSeed)
    return smote_si

def WFResample_si(biasToUniformClass=0.0,
                  invertSelection=False,
                  noReplacement=False,
                  randomSeed=1,
                  sampleSizePercent=100.0):
    resample_si = Resample_si()
    if not biasToUniformClass is None: resample_si.setBiasToUniformClass(biasToUniformClass)
    if not invertSelection is None: resample_si.setInvertSelection(invertSelection)
    if not noReplacement is None: resample_si.setNoReplacement(noReplacement)
    if not randomSeed is None: resample_si.setRandomSeed(randomSeed)
    if not sampleSizePercent is None: resample_si.setSampleSizePercent(sampleSizePercent)
    return resample_si

def WFSpreadSubsample_si(adjustWeights=False,
                         distributionSpread=0.0,
                         maxCount=0.0,
                         randomSeed=1):
    spreadsubsample_si = SpreadSubsample_si()
    if not adjustWeights is None: spreadsubsample_si.setAdjustWeights(adjustWeights)
    if not distributionSpread is None: spreadsubsample_si.setDistributionSpread(distributionSpread)
    if not maxCount is None: spreadsubsample_si.setMaxCount(maxCount)
    if not randomSeed is None: spreadsubsample_si.setRandomSeed(randomSeed)
    return spreadsubsample_si

def WFDiscretize_sa(attributeIndices='first-last',
                    invertSelection=False,
                    makeBinary=False,
                    useBetterEncoding=False,
                    useKononenko=False):
    discretize_sa = Discretize_sa()
    if not attributeIndices is None: discretize_sa.setAttributeIndices(attributeIndices)
    if not invertSelection is None: discretize_sa.setInvertSelection(invertSelection)
    if not makeBinary is None: discretize_sa.setMakeBinary(makeBinary)
    if not useBetterEncoding is None: discretize_sa.setUseBetterEncoding(useBetterEncoding)
    if not useKononenko is None: discretize_sa.setUseKononenko(useKononenko)
    return discretize_sa

def WFAddClassification_sa(classifier=None,
                           debug=False,
                           outputClassification=False,
                           outputDistribution=False,
                           outputErrorFlag=False,
                           removeOldClass=False,
                           serializedClassifierFile=None):
    addclassification_sa = AddClassification_sa()
    if not classifier is None: addclassification_sa.setClassifier(classifier)
    if not debug is None: addclassification_sa.setDebug(debug)
    if not outputClassification is None: addclassification_sa.setOutputClassification(outputClassification)
    if not outputDistribution is None: addclassification_sa.setOutputDistribution(outputDistribution)
    if not outputErrorFlag is None: addclassification_sa.setOutputErrorFlag(outputErrorFlag)
    if not removeOldClass is None: addclassification_sa.setRemoveOldClass(removeOldClass)
    if not serializedClassifierFile is None: addclassification_sa.setSerializedClassifierFile(serializedClassifierFile)
    return addclassification_sa

def WFNominalToBinary_sa(binaryAttributesNominal=False,
                         transformAllValues=False):
    nominaltobinary_sa = NominalToBinary_sa()
    if not binaryAttributesNominal is None: nominaltobinary_sa.setBinaryAttributesNominal(binaryAttributesNominal)
    if not transformAllValues is None: nominaltobinary_sa.setTransformAllValues(transformAllValues)
    return nominaltobinary_sa

def WFAttributeSelection_sa(evaluator=None,
                            search=None):
    attributeselection_sa = AttributeSelection_sa()
    if not evaluator is None: attributeselection_sa.setEvaluator(evaluator)
    if not search is None: attributeselection_sa.setSearch(search)
    return attributeselection_sa

def WFPLSFilter_sa(algorithm=None,
                   debug=False,
                   numComponents=20,
                   performPrediction=False,
                   preprocessing=None,
                   replaceMissing=True):
    plsfilter_sa = PLSFilter_sa()
    if not algorithm is None: plsfilter_sa.setAlgorithm(algorithm)
    if not debug is None: plsfilter_sa.setDebug(debug)
    if not numComponents is None: plsfilter_sa.setNumComponents(numComponents)
    if not performPrediction is None: plsfilter_sa.setPerformPrediction(performPrediction)
    if not preprocessing is None: plsfilter_sa.setPreprocessing(preprocessing)
    if not replaceMissing is None: plsfilter_sa.setReplaceMissing(replaceMissing)
    return plsfilter_sa

def WFClassOrder_sa(classOrder=0,
                    seed=1):
    classorder_sa = ClassOrder_sa()
    if not classOrder is None: classorder_sa.setClassOrder(classOrder)
    if not seed is None: classorder_sa.setSeed(seed)
    return classorder_sa

def WFSubsetByExpression_ui(debug=False,
                            expression='true',
                            filterAfterFirstBatch=False):
    subsetbyexpression_ui = SubsetByExpression_ui()
    if not debug is None: subsetbyexpression_ui.setDebug(debug)
    if not expression is None: subsetbyexpression_ui.setExpression(expression)
    if not filterAfterFirstBatch is None: subsetbyexpression_ui.setFilterAfterFirstBatch(filterAfterFirstBatch)
    return subsetbyexpression_ui

def WFNonSparseToSparse_ui(insertDummyNominalFirstValue=False,
                           treatMissingValuesAsZero=False):
    nonsparsetosparse_ui = NonSparseToSparse_ui()
    if not insertDummyNominalFirstValue is None: nonsparsetosparse_ui.setInsertDummyNominalFirstValue(insertDummyNominalFirstValue)
    if not treatMissingValuesAsZero is None: nonsparsetosparse_ui.setTreatMissingValuesAsZero(treatMissingValuesAsZero)
    return nonsparsetosparse_ui

def WFRemovePercentage_ui(invertSelection=False,
                          percentage=50.0):
    removepercentage_ui = RemovePercentage_ui()
    if not invertSelection is None: removepercentage_ui.setInvertSelection(invertSelection)
    if not percentage is None: removepercentage_ui.setPercentage(percentage)
    return removepercentage_ui

def WFSparseToNonSparse_ui():
    sparsetononsparse_ui = SparseToNonSparse_ui()
    return sparsetononsparse_ui

def WFDenormalize_ui(aggregationType=None,
                     groupingAttribute='first',
                     useOldMarketBasketFormat=False,
                     useSparseFormat=True):
    denormalize_ui = Denormalize_ui()
    if not aggregationType is None: denormalize_ui.setAggregationType(aggregationType)
    if not groupingAttribute is None: denormalize_ui.setGroupingAttribute(groupingAttribute)
    if not useOldMarketBasketFormat is None: denormalize_ui.setUseOldMarketBasketFormat(useOldMarketBasketFormat)
    if not useSparseFormat is None: denormalize_ui.setUseSparseFormat(useSparseFormat)
    return denormalize_ui

def WFRemoveFolds_ui(fold=1,
                     invertSelection=False,
                     numFolds=10,
                     seed=0):
    removefolds_ui = RemoveFolds_ui()
    if not fold is None: removefolds_ui.setFold(fold)
    if not invertSelection is None: removefolds_ui.setInvertSelection(invertSelection)
    if not numFolds is None: removefolds_ui.setNumFolds(numFolds)
    if not seed is None: removefolds_ui.setSeed(seed)
    return removefolds_ui

def WFNormalize_ui(LNorm=2.0,
                   norm=1.0):
    normalize_ui = Normalize_ui()
    if not LNorm is None: normalize_ui.setLNorm(LNorm)
    if not norm is None: normalize_ui.setNorm(norm)
    return normalize_ui

def WFRemoveWithValues_ui(attributeIndex='last',
                          dontFilterAfterFirstBatch=False,
                          invertSelection=False,
                          matchMissingValues=False,
                          modifyHeader=False,
                          nominalIndices='first-last',
                          splitPoint=0.0):
    removewithvalues_ui = RemoveWithValues_ui()
    if not attributeIndex is None: removewithvalues_ui.setAttributeIndex(attributeIndex)
    if not dontFilterAfterFirstBatch is None: removewithvalues_ui.setDontFilterAfterFirstBatch(dontFilterAfterFirstBatch)
    if not invertSelection is None: removewithvalues_ui.setInvertSelection(invertSelection)
    if not matchMissingValues is None: removewithvalues_ui.setMatchMissingValues(matchMissingValues)
    if not modifyHeader is None: removewithvalues_ui.setModifyHeader(modifyHeader)
    if not nominalIndices is None: removewithvalues_ui.setNominalIndices(nominalIndices)
    if not splitPoint is None: removewithvalues_ui.setSplitPoint(splitPoint)
    return removewithvalues_ui

def WFRandomize_ui(randomSeed=42):
    randomize_ui = Randomize_ui()
    if not randomSeed is None: randomize_ui.setRandomSeed(randomSeed)
    return randomize_ui

def WFRemoveMisclassified_ui(classIndex=-1,
                             classifier=None,
                             invert=False,
                             maxIterations=0,
                             numFolds=0,
                             threshold=0.1):
    removemisclassified_ui = RemoveMisclassified_ui()
    if not classIndex is None: removemisclassified_ui.setClassIndex(classIndex)
    if not classifier is None: removemisclassified_ui.setClassifier(classifier)
    if not invert is None: removemisclassified_ui.setInvert(invert)
    if not maxIterations is None: removemisclassified_ui.setMaxIterations(maxIterations)
    if not numFolds is None: removemisclassified_ui.setNumFolds(numFolds)
    if not threshold is None: removemisclassified_ui.setThreshold(threshold)
    return removemisclassified_ui

def WFRemoveRange_ui(instancesIndices='first-last',
                     invertSelection=False):
    removerange_ui = RemoveRange_ui()
    if not instancesIndices is None: removerange_ui.setInstancesIndices(instancesIndices)
    if not invertSelection is None: removerange_ui.setInvertSelection(invertSelection)
    return removerange_ui

def WFResample_ui(invertSelection=False,
                  noReplacement=False,
                  randomSeed=1,
                  sampleSizePercent=100.0):
    resample_ui = Resample_ui()
    if not invertSelection is None: resample_ui.setInvertSelection(invertSelection)
    if not noReplacement is None: resample_ui.setNoReplacement(noReplacement)
    if not randomSeed is None: resample_ui.setRandomSeed(randomSeed)
    if not sampleSizePercent is None: resample_ui.setSampleSizePercent(sampleSizePercent)
    return resample_ui

def WFRemoveFrequentValues_ui(attributeIndex='last',
                              invertSelection=False,
                              modifyHeader=False,
                              numValues=2,
                              useLeastValues=False):
    removefrequentvalues_ui = RemoveFrequentValues_ui()
    if not attributeIndex is None: removefrequentvalues_ui.setAttributeIndex(attributeIndex)
    if not invertSelection is None: removefrequentvalues_ui.setInvertSelection(invertSelection)
    if not modifyHeader is None: removefrequentvalues_ui.setModifyHeader(modifyHeader)
    if not numValues is None: removefrequentvalues_ui.setNumValues(numValues)
    if not useLeastValues is None: removefrequentvalues_ui.setUseLeastValues(useLeastValues)
    return removefrequentvalues_ui

def WFReservoirSample_ui(randomSeed=1,
                         sampleSize=100):
    reservoirsample_ui = ReservoirSample_ui()
    if not randomSeed is None: reservoirsample_ui.setRandomSeed(randomSeed)
    if not sampleSize is None: reservoirsample_ui.setSampleSize(sampleSize)
    return reservoirsample_ui

def WFDiscretize_ua(attributeIndices='first-last',
                    bins=10,
                    desiredWeightOfInstancesPerInterval=-1.0,
                    findNumBins=False,
                    ignoreClass=False,
                    invertSelection=False,
                    makeBinary=False,
                    useEqualFrequency=False):
    discretize_ua = Discretize_ua()
    if not attributeIndices is None: discretize_ua.setAttributeIndices(attributeIndices)
    if not bins is None: discretize_ua.setBins(bins)
    if not desiredWeightOfInstancesPerInterval is None: discretize_ua.setDesiredWeightOfInstancesPerInterval(desiredWeightOfInstancesPerInterval)
    if not findNumBins is None: discretize_ua.setFindNumBins(findNumBins)
    if not ignoreClass is None: discretize_ua.setIgnoreClass(ignoreClass)
    if not invertSelection is None: discretize_ua.setInvertSelection(invertSelection)
    if not makeBinary is None: discretize_ua.setMakeBinary(makeBinary)
    if not useEqualFrequency is None: discretize_ua.setUseEqualFrequency(useEqualFrequency)
    return discretize_ua

def WFNumericCleaner_ua(attributeIndices='first-last',
                        closeTo=0.0,
                        closeToDefault=0.0,
                        closeToTolerance=1e-06,
                        debug=False,
                        decimals=-1,
                        includeClass=False,
                        invertSelection=False,
                        maxDefault=1.79769313486e+308,
                        maxThreshold=1.79769313486e+308,
                        minDefault=-1.79769313486e+308,
                        minThreshold=-1.79769313486e+308):
    numericcleaner_ua = NumericCleaner_ua()
    if not attributeIndices is None: numericcleaner_ua.setAttributeIndices(attributeIndices)
    if not closeTo is None: numericcleaner_ua.setCloseTo(closeTo)
    if not closeToDefault is None: numericcleaner_ua.setCloseToDefault(closeToDefault)
    if not closeToTolerance is None: numericcleaner_ua.setCloseToTolerance(closeToTolerance)
    if not debug is None: numericcleaner_ua.setDebug(debug)
    if not decimals is None: numericcleaner_ua.setDecimals(decimals)
    if not includeClass is None: numericcleaner_ua.setIncludeClass(includeClass)
    if not invertSelection is None: numericcleaner_ua.setInvertSelection(invertSelection)
    if not maxDefault is None: numericcleaner_ua.setMaxDefault(maxDefault)
    if not maxThreshold is None: numericcleaner_ua.setMaxThreshold(maxThreshold)
    if not minDefault is None: numericcleaner_ua.setMinDefault(minDefault)
    if not minThreshold is None: numericcleaner_ua.setMinThreshold(minThreshold)
    return numericcleaner_ua

def WFWavelet_ua(algorithm=None,
                 debug=False,
                 filter=None,
                 padding=None):
    wavelet_ua = Wavelet_ua()
    if not algorithm is None: wavelet_ua.setAlgorithm(algorithm)
    if not debug is None: wavelet_ua.setDebug(debug)
    if not filter is None: wavelet_ua.setFilter(filter)
    if not padding is None: wavelet_ua.setPadding(padding)
    return wavelet_ua

def WFCenter_ua(ignoreClass=False):
    center_ua = Center_ua()
    if not ignoreClass is None: center_ua.setIgnoreClass(ignoreClass)
    return center_ua

def WFNominalToString_ua(attributeIndexes='last'):
    nominaltostring_ua = NominalToString_ua()
    if not attributeIndexes is None: nominaltostring_ua.setAttributeIndexes(attributeIndexes)
    return nominaltostring_ua

def WFEMImputation_ua(debug=False,
                      logLikelihoodThreshold=0.0001,
                      numIterations=-1,
                      ridge=1e-08,
                      useRidgePrior=False):
    emimputation_ua = EMImputation_ua()
    if not debug is None: emimputation_ua.setDebug(debug)
    if not logLikelihoodThreshold is None: emimputation_ua.setLogLikelihoodThreshold(logLikelihoodThreshold)
    if not numIterations is None: emimputation_ua.setNumIterations(numIterations)
    if not ridge is None: emimputation_ua.setRidge(ridge)
    if not useRidgePrior is None: emimputation_ua.setUseRidgePrior(useRidgePrior)
    return emimputation_ua

def WFStandardize_ua(ignoreClass=False):
    standardize_ua = Standardize_ua()
    if not ignoreClass is None: standardize_ua.setIgnoreClass(ignoreClass)
    return standardize_ua

def WFTimeSeriesTranslate_ua(attributeIndices='',
                             fillWithMissing=True,
                             instanceRange=-1,
                             invertSelection=False):
    timeseriestranslate_ua = TimeSeriesTranslate_ua()
    if not attributeIndices is None: timeseriestranslate_ua.setAttributeIndices(attributeIndices)
    if not fillWithMissing is None: timeseriestranslate_ua.setFillWithMissing(fillWithMissing)
    if not instanceRange is None: timeseriestranslate_ua.setInstanceRange(instanceRange)
    if not invertSelection is None: timeseriestranslate_ua.setInvertSelection(invertSelection)
    return timeseriestranslate_ua

def WFTimeSeriesDelta_ua(attributeIndices='',
                         fillWithMissing=True,
                         instanceRange=-1,
                         invertSelection=False):
    timeseriesdelta_ua = TimeSeriesDelta_ua()
    if not attributeIndices is None: timeseriesdelta_ua.setAttributeIndices(attributeIndices)
    if not fillWithMissing is None: timeseriesdelta_ua.setFillWithMissing(fillWithMissing)
    if not instanceRange is None: timeseriesdelta_ua.setInstanceRange(instanceRange)
    if not invertSelection is None: timeseriesdelta_ua.setInvertSelection(invertSelection)
    return timeseriesdelta_ua

def WFNominalToBinary_ua(attributeIndices='first-last',
                         binaryAttributesNominal=False,
                         invertSelection=False,
                         transformAllValues=False):
    nominaltobinary_ua = NominalToBinary_ua()
    if not attributeIndices is None: nominaltobinary_ua.setAttributeIndices(attributeIndices)
    if not binaryAttributesNominal is None: nominaltobinary_ua.setBinaryAttributesNominal(binaryAttributesNominal)
    if not invertSelection is None: nominaltobinary_ua.setInvertSelection(invertSelection)
    if not transformAllValues is None: nominaltobinary_ua.setTransformAllValues(transformAllValues)
    return nominaltobinary_ua

def WFReplaceMissingValues_ua(ignoreClass=False):
    replacemissingvalues_ua = ReplaceMissingValues_ua()
    if not ignoreClass is None: replacemissingvalues_ua.setIgnoreClass(ignoreClass)
    return replacemissingvalues_ua

def WFPartitionedMultiFilter_ua(debug=False,
                                filters=None,
                                ranges=None,
                                removeUnused=False):
    partitionedmultifilter_ua = PartitionedMultiFilter_ua()
    if not debug is None: partitionedmultifilter_ua.setDebug(debug)
    if not filters is None: partitionedmultifilter_ua.setFilters(filters)
    if not ranges is None: partitionedmultifilter_ua.setRanges(ranges)
    if not removeUnused is None: partitionedmultifilter_ua.setRemoveUnused(removeUnused)
    return partitionedmultifilter_ua

def WFAddValues_ua(attributeIndex='last',
                   labels='',
                   sort=False):
    addvalues_ua = AddValues_ua()
    if not attributeIndex is None: addvalues_ua.setAttributeIndex(attributeIndex)
    if not labels is None: addvalues_ua.setLabels(labels)
    if not sort is None: addvalues_ua.setSort(sort)
    return addvalues_ua

def WFAddCluster_ua(clusterer=None,
                    ignoredAttributeIndices='',
                    serializedClustererFile=None):
    addcluster_ua = AddCluster_ua()
    if not clusterer is None: addcluster_ua.setClusterer(clusterer)
    if not ignoredAttributeIndices is None: addcluster_ua.setIgnoredAttributeIndices(ignoredAttributeIndices)
    if not serializedClustererFile is None: addcluster_ua.setSerializedClustererFile(serializedClustererFile)
    return addcluster_ua

def WFRemoveType_ua(attributeType=None,
                    invertSelection=False):
    removetype_ua = RemoveType_ua()
    if not attributeType is None: removetype_ua.setAttributeType(attributeType)
    if not invertSelection is None: removetype_ua.setInvertSelection(invertSelection)
    return removetype_ua

def WFReorder_ua(attributeIndices='first-last'):
    reorder_ua = Reorder_ua()
    if not attributeIndices is None: reorder_ua.setAttributeIndices(attributeIndices)
    return reorder_ua

def WFSortLabels_ua(debug=False,
                    invertSelection=False,
                    sortType=None):
    sortlabels_ua = SortLabels_ua()
    if not debug is None: sortlabels_ua.setDebug(debug)
    if not invertSelection is None: sortlabels_ua.setInvertSelection(invertSelection)
    if not sortType is None: sortlabels_ua.setSortType(sortType)
    return sortlabels_ua

def WFNormalize_ua(ignoreClass=False,
                   scale=1.0,
                   translation=0.0):
    normalize_ua = Normalize_ua()
    if not ignoreClass is None: normalize_ua.setIgnoreClass(ignoreClass)
    if not scale is None: normalize_ua.setScale(scale)
    if not translation is None: normalize_ua.setTranslation(translation)
    return normalize_ua

def WFMakeIndicator_ua(attributeIndex='last',
                       numeric=True,
                       valueIndices='last'):
    makeindicator_ua = MakeIndicator_ua()
    if not attributeIndex is None: makeindicator_ua.setAttributeIndex(attributeIndex)
    if not numeric is None: makeindicator_ua.setNumeric(numeric)
    if not valueIndices is None: makeindicator_ua.setValueIndices(valueIndices)
    return makeindicator_ua

def WFSwapValues_ua(attributeIndex='last',
                    firstValueIndex='first',
                    secondValueIndex='last'):
    swapvalues_ua = SwapValues_ua()
    if not attributeIndex is None: swapvalues_ua.setAttributeIndex(attributeIndex)
    if not firstValueIndex is None: swapvalues_ua.setFirstValueIndex(firstValueIndex)
    if not secondValueIndex is None: swapvalues_ua.setSecondValueIndex(secondValueIndex)
    return swapvalues_ua

def WFInterquartileRange_ua(attributeIndices='first-last',
                            debug=False,
                            detectionPerAttribute=False,
                            extremeValuesAsOutliers=False,
                            extremeValuesFactor=6.0,
                            outlierFactor=3.0,
                            outputOffsetMultiplier=False):
    interquartilerange_ua = InterquartileRange_ua()
    if not attributeIndices is None: interquartilerange_ua.setAttributeIndices(attributeIndices)
    if not debug is None: interquartilerange_ua.setDebug(debug)
    if not detectionPerAttribute is None: interquartilerange_ua.setDetectionPerAttribute(detectionPerAttribute)
    if not extremeValuesAsOutliers is None: interquartilerange_ua.setExtremeValuesAsOutliers(extremeValuesAsOutliers)
    if not extremeValuesFactor is None: interquartilerange_ua.setExtremeValuesFactor(extremeValuesFactor)
    if not outlierFactor is None: interquartilerange_ua.setOutlierFactor(outlierFactor)
    if not outputOffsetMultiplier is None: interquartilerange_ua.setOutputOffsetMultiplier(outputOffsetMultiplier)
    return interquartilerange_ua

def WFRemove_ua(attributeIndices='',
                invertSelection=False):
    remove_ua = Remove_ua()
    if not attributeIndices is None: remove_ua.setAttributeIndices(attributeIndices)
    if not invertSelection is None: remove_ua.setInvertSelection(invertSelection)
    return remove_ua

def WFPrincipalComponents_ua(centerData=False,
                             maximumAttributeNames=5,
                             maximumAttributes=-1,
                             varianceCovered=0.95):
    principalcomponents_ua = PrincipalComponents_ua()
    if not centerData is None: principalcomponents_ua.setCenterData(centerData)
    if not maximumAttributeNames is None: principalcomponents_ua.setMaximumAttributeNames(maximumAttributeNames)
    if not maximumAttributes is None: principalcomponents_ua.setMaximumAttributes(maximumAttributes)
    if not varianceCovered is None: principalcomponents_ua.setVarianceCovered(varianceCovered)
    return principalcomponents_ua

def WFClassAssigner_ua(classIndex='last',
                       debug=False):
    classassigner_ua = ClassAssigner_ua()
    if not classIndex is None: classassigner_ua.setClassIndex(classIndex)
    if not debug is None: classassigner_ua.setDebug(debug)
    return classassigner_ua

def WFAddExpression_ua(debug=False,
                       expression='a1^2',
                       name='expression'):
    addexpression_ua = AddExpression_ua()
    if not debug is None: addexpression_ua.setDebug(debug)
    if not expression is None: addexpression_ua.setExpression(expression)
    if not name is None: addexpression_ua.setName(name)
    return addexpression_ua

def WFNumericTransform_ua(attributeIndices='',
                          className='java.lang.Math',
                          invertSelection=False,
                          methodName='abs'):
    numerictransform_ua = NumericTransform_ua()
    if not attributeIndices is None: numerictransform_ua.setAttributeIndices(attributeIndices)
    if not className is None: numerictransform_ua.setClassName(className)
    if not invertSelection is None: numerictransform_ua.setInvertSelection(invertSelection)
    if not methodName is None: numerictransform_ua.setMethodName(methodName)
    return numerictransform_ua

def WFMathExpression_ua(expression='(A-MIN)/(MAX-MIN)',
                        ignoreClass=False,
                        ignoreRange='',
                        invertSelection=False):
    mathexpression_ua = MathExpression_ua()
    if not expression is None: mathexpression_ua.setExpression(expression)
    if not ignoreClass is None: mathexpression_ua.setIgnoreClass(ignoreClass)
    if not ignoreRange is None: mathexpression_ua.setIgnoreRange(ignoreRange)
    if not invertSelection is None: mathexpression_ua.setInvertSelection(invertSelection)
    return mathexpression_ua

def WFNumericToBinary_ua(ignoreClass=False):
    numerictobinary_ua = NumericToBinary_ua()
    if not ignoreClass is None: numerictobinary_ua.setIgnoreClass(ignoreClass)
    return numerictobinary_ua

def WFPropositionalToMultiInstance_ua(randomize=False,
                                      seed=1):
    propositionaltomultiinstance_ua = PropositionalToMultiInstance_ua()
    if not randomize is None: propositionaltomultiinstance_ua.setRandomize(randomize)
    if not seed is None: propositionaltomultiinstance_ua.setSeed(seed)
    return propositionaltomultiinstance_ua

def WFRandomProjection_ua(distribution=None,
                          numberOfAttributes=10,
                          percent=0.0,
                          randomSeed=42,
                          replaceMissingValues=False):
    randomprojection_ua = RandomProjection_ua()
    if not distribution is None: randomprojection_ua.setDistribution(distribution)
    if not numberOfAttributes is None: randomprojection_ua.setNumberOfAttributes(numberOfAttributes)
    if not percent is None: randomprojection_ua.setPercent(percent)
    if not randomSeed is None: randomprojection_ua.setRandomSeed(randomSeed)
    if not replaceMissingValues is None: randomprojection_ua.setReplaceMissingValues(replaceMissingValues)
    return randomprojection_ua

def WFAddNoise_ua(attributeIndex='last',
                  percent=10,
                  randomSeed=1,
                  useMissing=False):
    addnoise_ua = AddNoise_ua()
    if not attributeIndex is None: addnoise_ua.setAttributeIndex(attributeIndex)
    if not percent is None: addnoise_ua.setPercent(percent)
    if not randomSeed is None: addnoise_ua.setRandomSeed(randomSeed)
    if not useMissing is None: addnoise_ua.setUseMissing(useMissing)
    return addnoise_ua

def WFFirstOrder_ua(attributeIndices=''):
    firstorder_ua = FirstOrder_ua()
    if not attributeIndices is None: firstorder_ua.setAttributeIndices(attributeIndices)
    return firstorder_ua

def WFMILESFilter_ua(debug=False,
                     sigma=894.427191):
    milesfilter_ua = MILESFilter_ua()
    if not debug is None: milesfilter_ua.setDebug(debug)
    if not sigma is None: milesfilter_ua.setSigma(sigma)
    return milesfilter_ua

def WFRandomSubset_ua(debug=False,
                      numAttributes=0.5,
                      seed=1):
    randomsubset_ua = RandomSubset_ua()
    if not debug is None: randomsubset_ua.setDebug(debug)
    if not numAttributes is None: randomsubset_ua.setNumAttributes(numAttributes)
    if not seed is None: randomsubset_ua.setSeed(seed)
    return randomsubset_ua

def WFKernelFilter_ua(checksTurnedOff=False,
                      debug=False,
                      initFile=None,
                      initFileClassIndex='last',
                      kernel=None,
                      kernelFactorExpression='1',
                      preprocessing=None):
    kernelfilter_ua = KernelFilter_ua()
    if not checksTurnedOff is None: kernelfilter_ua.setChecksTurnedOff(checksTurnedOff)
    if not debug is None: kernelfilter_ua.setDebug(debug)
    if not initFile is None: kernelfilter_ua.setInitFile(initFile)
    if not initFileClassIndex is None: kernelfilter_ua.setInitFileClassIndex(initFileClassIndex)
    if not kernel is None: kernelfilter_ua.setKernel(kernel)
    if not kernelFactorExpression is None: kernelfilter_ua.setKernelFactorExpression(kernelFactorExpression)
    if not preprocessing is None: kernelfilter_ua.setPreprocessing(preprocessing)
    return kernelfilter_ua

def WFMergeTwoValues_ua(attributeIndex='last',
                        firstValueIndex='first',
                        secondValueIndex='last'):
    mergetwovalues_ua = MergeTwoValues_ua()
    if not attributeIndex is None: mergetwovalues_ua.setAttributeIndex(attributeIndex)
    if not firstValueIndex is None: mergetwovalues_ua.setFirstValueIndex(firstValueIndex)
    if not secondValueIndex is None: mergetwovalues_ua.setSecondValueIndex(secondValueIndex)
    return mergetwovalues_ua

def WFClusterMembership_ua(densityBasedClusterer=None,
                           ignoredAttributeIndices=''):
    clustermembership_ua = ClusterMembership_ua()
    if not densityBasedClusterer is None: clustermembership_ua.setDensityBasedClusterer(densityBasedClusterer)
    if not ignoredAttributeIndices is None: clustermembership_ua.setIgnoredAttributeIndices(ignoredAttributeIndices)
    return clustermembership_ua

def WFAddID_ua(IDIndex='first',
               attributeName='ID'):
    addid_ua = AddID_ua()
    if not IDIndex is None: addid_ua.setIDIndex(IDIndex)
    if not attributeName is None: addid_ua.setAttributeName(attributeName)
    return addid_ua

def WFMultiInstanceToPropositional_ua(weightMethod=None):
    multiinstancetopropositional_ua = MultiInstanceToPropositional_ua()
    if not weightMethod is None: multiinstancetopropositional_ua.setWeightMethod(weightMethod)
    return multiinstancetopropositional_ua

def WFStringToNominal_ua(attributeRange='last'):
    stringtonominal_ua = StringToNominal_ua()
    if not attributeRange is None: stringtonominal_ua.setAttributeRange(attributeRange)
    return stringtonominal_ua

def WFRemoveUseless_ua(maximumVariancePercentageAllowed=99.0):
    removeuseless_ua = RemoveUseless_ua()
    if not maximumVariancePercentageAllowed is None: removeuseless_ua.setMaximumVariancePercentageAllowed(maximumVariancePercentageAllowed)
    return removeuseless_ua

def WFRenameAttribute_ua(attributeIndices='first-last',
                         debug=False,
                         find='([\s\S]+)',
                         invertSelection=False,
                         replace='$0',
                         replaceAll=False):
    renameattribute_ua = RenameAttribute_ua()
    if not attributeIndices is None: renameattribute_ua.setAttributeIndices(attributeIndices)
    if not debug is None: renameattribute_ua.setDebug(debug)
    if not find is None: renameattribute_ua.setFind(find)
    if not invertSelection is None: renameattribute_ua.setInvertSelection(invertSelection)
    if not replace is None: renameattribute_ua.setReplace(replace)
    if not replaceAll is None: renameattribute_ua.setReplaceAll(replaceAll)
    return renameattribute_ua

def WFMergeManyValues_ua(attributeIndex='last',
                         label='merged',
                         mergeValueRange='1,2'):
    mergemanyvalues_ua = MergeManyValues_ua()
    if not attributeIndex is None: mergemanyvalues_ua.setAttributeIndex(attributeIndex)
    if not label is None: mergemanyvalues_ua.setLabel(label)
    if not mergeValueRange is None: mergemanyvalues_ua.setMergeValueRange(mergeValueRange)
    return mergemanyvalues_ua

def WFObfuscate_ua():
    obfuscate_ua = Obfuscate_ua()
    return obfuscate_ua

def WFRELAGGS_ua(debug=False,
                 invertSelection=False,
                 maxCardinality=20):
    relaggs_ua = RELAGGS_ua()
    if not debug is None: relaggs_ua.setDebug(debug)
    if not invertSelection is None: relaggs_ua.setInvertSelection(invertSelection)
    if not maxCardinality is None: relaggs_ua.setMaxCardinality(maxCardinality)
    return relaggs_ua

def WFNumericToNominal_ua(attributeIndices='first-last',
                          debug=False,
                          invertSelection=False):
    numerictonominal_ua = NumericToNominal_ua()
    if not attributeIndices is None: numerictonominal_ua.setAttributeIndices(attributeIndices)
    if not debug is None: numerictonominal_ua.setDebug(debug)
    if not invertSelection is None: numerictonominal_ua.setInvertSelection(invertSelection)
    return numerictonominal_ua

def WFCopy_ua(attributeIndices='',
              invertSelection=False):
    copy_ua = Copy_ua()
    if not attributeIndices is None: copy_ua.setAttributeIndices(attributeIndices)
    if not invertSelection is None: copy_ua.setInvertSelection(invertSelection)
    return copy_ua

def WFPKIDiscretize_ua(attributeIndices='first-last',
                       bins=0,
                       desiredWeightOfInstancesPerInterval=-1.0,
                       findNumBins=False,
                       ignoreClass=False,
                       invertSelection=False,
                       makeBinary=False,
                       useEqualFrequency=True):
    pkidiscretize_ua = PKIDiscretize_ua()
    if not attributeIndices is None: pkidiscretize_ua.setAttributeIndices(attributeIndices)
    if not bins is None: pkidiscretize_ua.setBins(bins)
    if not desiredWeightOfInstancesPerInterval is None: pkidiscretize_ua.setDesiredWeightOfInstancesPerInterval(desiredWeightOfInstancesPerInterval)
    if not findNumBins is None: pkidiscretize_ua.setFindNumBins(findNumBins)
    if not ignoreClass is None: pkidiscretize_ua.setIgnoreClass(ignoreClass)
    if not invertSelection is None: pkidiscretize_ua.setInvertSelection(invertSelection)
    if not makeBinary is None: pkidiscretize_ua.setMakeBinary(makeBinary)
    if not useEqualFrequency is None: pkidiscretize_ua.setUseEqualFrequency(useEqualFrequency)
    return pkidiscretize_ua

def WFRemoveByName_ua(debug=False,
                      expression='^.*id$',
                      invertSelection=False):
    removebyname_ua = RemoveByName_ua()
    if not debug is None: removebyname_ua.setDebug(debug)
    if not expression is None: removebyname_ua.setExpression(expression)
    if not invertSelection is None: removebyname_ua.setInvertSelection(invertSelection)
    return removebyname_ua