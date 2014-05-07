from jpype import *
from oscail.common.integration.jpype_utils import jpype_bootup

jpype_bootup()

S3F=JClass('weka.attributeSelection.S3F')
ReliefFAttributeEval=JClass('weka.attributeSelection.ReliefFAttributeEval')
WrapperSubsetEval=JClass('weka.attributeSelection.WrapperSubsetEval')
ChiSquaredAttributeEval=JClass('weka.attributeSelection.ChiSquaredAttributeEval')
GainRatioAttributeEval=JClass('weka.attributeSelection.GainRatioAttributeEval')
ConsistencySubsetEval=JClass('weka.attributeSelection.ConsistencySubsetEval')
PrincipalComponents=JClass('weka.attributeSelection.PrincipalComponents')
CfsSubsetEval=JClass('weka.attributeSelection.CfsSubsetEval')
SymmetricalUncertAttributeSetEval=JClass('weka.attributeSelection.SymmetricalUncertAttributeSetEval')
LatentSemanticAnalysis=JClass('weka.attributeSelection.LatentSemanticAnalysis')
SVMAttributeEval=JClass('weka.attributeSelection.SVMAttributeEval')
FilteredSubsetEval=JClass('weka.attributeSelection.FilteredSubsetEval')
OneRAttributeEval=JClass('weka.attributeSelection.OneRAttributeEval')
CostSensitiveAttributeEval=JClass('weka.attributeSelection.CostSensitiveAttributeEval')
FilteredAttributeEval=JClass('weka.attributeSelection.FilteredAttributeEval')
ClassifierAttributeEval=JClass('weka.attributeSelection.ClassifierAttributeEval')
SignificanceAttributeEval=JClass('weka.attributeSelection.SignificanceAttributeEval')
InfoGainAttributeEval=JClass('weka.attributeSelection.InfoGainAttributeEval')
ClassifierSubsetEval=JClass('weka.attributeSelection.ClassifierSubsetEval')
CostSensitiveSubsetEval=JClass('weka.attributeSelection.CostSensitiveSubsetEval')
SSF=JClass('weka.attributeSelection.SSF')
SymmetricalUncertAttributeEval=JClass('weka.attributeSelection.SymmetricalUncertAttributeEval')
GeneticSearch=JClass('weka.attributeSelection.GeneticSearch')
RerankingSearch=JClass('weka.attributeSelection.RerankingSearch')
LinearForwardSelection=JClass('weka.attributeSelection.LinearForwardSelection')
BestFirst=JClass('weka.attributeSelection.BestFirst')
RankSearch=JClass('weka.attributeSelection.RankSearch')
PSOSearch=JClass('weka.attributeSelection.PSOSearch')
TabuSearch=JClass('weka.attributeSelection.TabuSearch')
SubsetSizeForwardSelection=JClass('weka.attributeSelection.SubsetSizeForwardSelection')
ExhaustiveSearch=JClass('weka.attributeSelection.ExhaustiveSearch')
RandomSearch=JClass('weka.attributeSelection.RandomSearch')
Ranker=JClass('weka.attributeSelection.Ranker')
FCBFSearch=JClass('weka.attributeSelection.FCBFSearch')
RaceSearch=JClass('weka.attributeSelection.RaceSearch')
ScatterSearchV1=JClass('weka.attributeSelection.ScatterSearchV1')
GreedyStepwise=JClass('weka.attributeSelection.GreedyStepwise')

def WASS3F(strategy=None):
    s3f = S3F()
    if not strategy is None: s3f.setStrategy(strategy)
    return s3f

def WASReliefFAttributeEval(numNeighbours=10,
                            sampleSize=-1,
                            seed=1,
                            sigma=2,
                            weightByDistance=False):
    relieffattributeeval = ReliefFAttributeEval()
    if not numNeighbours is None: relieffattributeeval.setNumNeighbours(numNeighbours)
    if not sampleSize is None: relieffattributeeval.setSampleSize(sampleSize)
    if not seed is None: relieffattributeeval.setSeed(seed)
    if not sigma is None: relieffattributeeval.setSigma(sigma)
    if not weightByDistance is None: relieffattributeeval.setWeightByDistance(weightByDistance)
    return relieffattributeeval

def WASWrapperSubsetEval(classifier=None,
                         evaluationMeasure=None,
                         folds=5,
                         seed=1,
                         threshold=0.01):
    wrappersubseteval = WrapperSubsetEval()
    if not classifier is None: wrappersubseteval.setClassifier(classifier)
    if not evaluationMeasure is None: wrappersubseteval.setEvaluationMeasure(evaluationMeasure)
    if not folds is None: wrappersubseteval.setFolds(folds)
    if not seed is None: wrappersubseteval.setSeed(seed)
    if not threshold is None: wrappersubseteval.setThreshold(threshold)
    return wrappersubseteval

def WASChiSquaredAttributeEval(binarizeNumericAttributes=False,
                               missingMerge=True):
    chisquaredattributeeval = ChiSquaredAttributeEval()
    if not binarizeNumericAttributes is None: chisquaredattributeeval.setBinarizeNumericAttributes(binarizeNumericAttributes)
    if not missingMerge is None: chisquaredattributeeval.setMissingMerge(missingMerge)
    return chisquaredattributeeval

def WASGainRatioAttributeEval(missingMerge=True):
    gainratioattributeeval = GainRatioAttributeEval()
    if not missingMerge is None: gainratioattributeeval.setMissingMerge(missingMerge)
    return gainratioattributeeval

def WASConsistencySubsetEval():
    consistencysubseteval = ConsistencySubsetEval()
    return consistencysubseteval

def WASPrincipalComponents(centerData=False,
                           maximumAttributeNames=5,
                           transformBackToOriginal=False,
                           varianceCovered=0.95):
    principalcomponents = PrincipalComponents()
    if not centerData is None: principalcomponents.setCenterData(centerData)
    if not maximumAttributeNames is None: principalcomponents.setMaximumAttributeNames(maximumAttributeNames)
    if not transformBackToOriginal is None: principalcomponents.setTransformBackToOriginal(transformBackToOriginal)
    if not varianceCovered is None: principalcomponents.setVarianceCovered(varianceCovered)
    return principalcomponents

def WASCfsSubsetEval(locallyPredictive=True,
                     missingSeparate=False):
    cfssubseteval = CfsSubsetEval()
    if not locallyPredictive is None: cfssubseteval.setLocallyPredictive(locallyPredictive)
    if not missingSeparate is None: cfssubseteval.setMissingSeparate(missingSeparate)
    return cfssubseteval

def WASSymmetricalUncertAttributeSetEval(missingMerge=True):
    symmetricaluncertattributeseteval = SymmetricalUncertAttributeSetEval()
    if not missingMerge is None: symmetricaluncertattributeseteval.setMissingMerge(missingMerge)
    return symmetricaluncertattributeseteval

def WASLatentSemanticAnalysis(maximumAttributeNames=5,
                              normalize=False,
                              rank=0.95):
    latentsemanticanalysis = LatentSemanticAnalysis()
    if not maximumAttributeNames is None: latentsemanticanalysis.setMaximumAttributeNames(maximumAttributeNames)
    if not normalize is None: latentsemanticanalysis.setNormalize(normalize)
    if not rank is None: latentsemanticanalysis.setRank(rank)
    return latentsemanticanalysis

def WASSVMAttributeEval(attsToEliminatePerIteration=1,
                        complexityParameter=1.0,
                        epsilonParameter=1e-25,
                        filterType=None,
                        percentThreshold=0,
                        percentToEliminatePerIteration=0,
                        toleranceParameter=1e-10):
    svmattributeeval = SVMAttributeEval()
    if not attsToEliminatePerIteration is None: svmattributeeval.setAttsToEliminatePerIteration(attsToEliminatePerIteration)
    if not complexityParameter is None: svmattributeeval.setComplexityParameter(complexityParameter)
    if not epsilonParameter is None: svmattributeeval.setEpsilonParameter(epsilonParameter)
    if not filterType is None: svmattributeeval.setFilterType(filterType)
    if not percentThreshold is None: svmattributeeval.setPercentThreshold(percentThreshold)
    if not percentToEliminatePerIteration is None: svmattributeeval.setPercentToEliminatePerIteration(percentToEliminatePerIteration)
    if not toleranceParameter is None: svmattributeeval.setToleranceParameter(toleranceParameter)
    return svmattributeeval

def WASFilteredSubsetEval(filter=None,
                          subsetEvaluator=None):
    filteredsubseteval = FilteredSubsetEval()
    if not filter is None: filteredsubseteval.setFilter(filter)
    if not subsetEvaluator is None: filteredsubseteval.setSubsetEvaluator(subsetEvaluator)
    return filteredsubseteval

def WASOneRAttributeEval(evalUsingTrainingData=False,
                         folds=10,
                         minimumBucketSize=6,
                         seed=1):
    onerattributeeval = OneRAttributeEval()
    if not evalUsingTrainingData is None: onerattributeeval.setEvalUsingTrainingData(evalUsingTrainingData)
    if not folds is None: onerattributeeval.setFolds(folds)
    if not minimumBucketSize is None: onerattributeeval.setMinimumBucketSize(minimumBucketSize)
    if not seed is None: onerattributeeval.setSeed(seed)
    return onerattributeeval

def WASCostSensitiveAttributeEval(costMatrix= 0
,
                                  costMatrixSource=None,
                                  evaluator=None,
                                  onDemandDirectory=None,
                                  seed=1):
    costsensitiveattributeeval = CostSensitiveAttributeEval()
    if not costMatrix is None: costsensitiveattributeeval.setCostMatrix(costMatrix)
    if not costMatrixSource is None: costsensitiveattributeeval.setCostMatrixSource(costMatrixSource)
    if not evaluator is None: costsensitiveattributeeval.setEvaluator(evaluator)
    if not onDemandDirectory is None: costsensitiveattributeeval.setOnDemandDirectory(onDemandDirectory)
    if not seed is None: costsensitiveattributeeval.setSeed(seed)
    return costsensitiveattributeeval

def WASFilteredAttributeEval(attributeEvaluator=None,
                             filter=None):
    filteredattributeeval = FilteredAttributeEval()
    if not attributeEvaluator is None: filteredattributeeval.setAttributeEvaluator(attributeEvaluator)
    if not filter is None: filteredattributeeval.setFilter(filter)
    return filteredattributeeval

def WASClassifierAttributeEval(classifier=None,
                               evalUsingTrainingData=False,
                               folds=10,
                               seed=1):
    classifierattributeeval = ClassifierAttributeEval()
    if not classifier is None: classifierattributeeval.setClassifier(classifier)
    if not evalUsingTrainingData is None: classifierattributeeval.setEvalUsingTrainingData(evalUsingTrainingData)
    if not folds is None: classifierattributeeval.setFolds(folds)
    if not seed is None: classifierattributeeval.setSeed(seed)
    return classifierattributeeval

def WASSignificanceAttributeEval(missingMerge=True):
    significanceattributeeval = SignificanceAttributeEval()
    if not missingMerge is None: significanceattributeeval.setMissingMerge(missingMerge)
    return significanceattributeeval

def WASInfoGainAttributeEval(binarizeNumericAttributes=False,
                             missingMerge=True):
    infogainattributeeval = InfoGainAttributeEval()
    if not binarizeNumericAttributes is None: infogainattributeeval.setBinarizeNumericAttributes(binarizeNumericAttributes)
    if not missingMerge is None: infogainattributeeval.setMissingMerge(missingMerge)
    return infogainattributeeval

def WASClassifierSubsetEval(classifier=None,
                            holdOutFile=None,
                            useTraining=True):
    classifiersubseteval = ClassifierSubsetEval()
    if not classifier is None: classifiersubseteval.setClassifier(classifier)
    if not holdOutFile is None: classifiersubseteval.setHoldOutFile(holdOutFile)
    if not useTraining is None: classifiersubseteval.setUseTraining(useTraining)
    return classifiersubseteval

def WASCostSensitiveSubsetEval(costMatrix= 0
,
                               costMatrixSource=None,
                               evaluator=None,
                               onDemandDirectory=None,
                               seed=1):
    costsensitivesubseteval = CostSensitiveSubsetEval()
    if not costMatrix is None: costsensitivesubseteval.setCostMatrix(costMatrix)
    if not costMatrixSource is None: costsensitivesubseteval.setCostMatrixSource(costMatrixSource)
    if not evaluator is None: costsensitivesubseteval.setEvaluator(evaluator)
    if not onDemandDirectory is None: costsensitivesubseteval.setOnDemandDirectory(onDemandDirectory)
    if not seed is None: costsensitivesubseteval.setSeed(seed)
    return costsensitivesubseteval

def WASSSF(strategy=None):
    ssf = SSF()
    if not strategy is None: ssf.setStrategy(strategy)
    return ssf

def WASSymmetricalUncertAttributeEval(missingMerge=True):
    symmetricaluncertattributeeval = SymmetricalUncertAttributeEval()
    if not missingMerge is None: symmetricaluncertattributeeval.setMissingMerge(missingMerge)
    return symmetricaluncertattributeeval

def WASGeneticSearch(crossoverProb=0.6,
                     maxGenerations=20,
                     mutationProb=0.033,
                     populationSize=20,
                     reportFrequency=20,
                     seed=1,
                     startSet=''):
    geneticsearch = GeneticSearch()
    if not crossoverProb is None: geneticsearch.setCrossoverProb(crossoverProb)
    if not maxGenerations is None: geneticsearch.setMaxGenerations(maxGenerations)
    if not mutationProb is None: geneticsearch.setMutationProb(mutationProb)
    if not populationSize is None: geneticsearch.setPopulationSize(populationSize)
    if not reportFrequency is None: geneticsearch.setReportFrequency(reportFrequency)
    if not seed is None: geneticsearch.setSeed(seed)
    if not startSet is None: geneticsearch.setStartSet(startSet)
    return geneticsearch

def WASRerankingSearch(b=20,
                       informationBasedEvaluator=None,
                       rerankMethod=None,
                       searchAlgorithm=None):
    rerankingsearch = RerankingSearch()
    if not b is None: rerankingsearch.setB(b)
    if not informationBasedEvaluator is None: rerankingsearch.setInformationBasedEvaluator(informationBasedEvaluator)
    if not rerankMethod is None: rerankingsearch.setRerankMethod(rerankMethod)
    if not searchAlgorithm is None: rerankingsearch.setSearchAlgorithm(searchAlgorithm)
    return rerankingsearch

def WASLinearForwardSelection(forwardSelectionMethod=None,
                              lookupCacheSize=1,
                              numUsedAttributes=50,
                              performRanking=True,
                              searchTermination=5,
                              startSet='',
                              type=None,
                              verbose=False):
    linearforwardselection = LinearForwardSelection()
    if not forwardSelectionMethod is None: linearforwardselection.setForwardSelectionMethod(forwardSelectionMethod)
    if not lookupCacheSize is None: linearforwardselection.setLookupCacheSize(lookupCacheSize)
    if not numUsedAttributes is None: linearforwardselection.setNumUsedAttributes(numUsedAttributes)
    if not performRanking is None: linearforwardselection.setPerformRanking(performRanking)
    if not searchTermination is None: linearforwardselection.setSearchTermination(searchTermination)
    if not startSet is None: linearforwardselection.setStartSet(startSet)
    if not type is None: linearforwardselection.setType(type)
    if not verbose is None: linearforwardselection.setVerbose(verbose)
    return linearforwardselection

def WASBestFirst(direction=None,
                 lookupCacheSize=1,
                 searchTermination=5,
                 startSet=''):
    bestfirst = BestFirst()
    if not direction is None: bestfirst.setDirection(direction)
    if not lookupCacheSize is None: bestfirst.setLookupCacheSize(lookupCacheSize)
    if not searchTermination is None: bestfirst.setSearchTermination(searchTermination)
    if not startSet is None: bestfirst.setStartSet(startSet)
    return bestfirst

def WASRankSearch(attributeEvaluator=None,
                  startPoint=0,
                  stepSize=1):
    ranksearch = RankSearch()
    if not attributeEvaluator is None: ranksearch.setAttributeEvaluator(attributeEvaluator)
    if not startPoint is None: ranksearch.setStartPoint(startPoint)
    if not stepSize is None: ranksearch.setStepSize(stepSize)
    return ranksearch

def WASPSOSearch(individualWeight=0.34,
                 inertiaWeight=0.33,
                 iterations=20,
                 logFile=None,
                 mutationProb=0.01,
                 mutationType=None,
                 populationSize=20,
                 reportFrequency=20,
                 seed=1,
                 socialWeight=0.33,
                 startSet=''):
    psosearch = PSOSearch()
    if not individualWeight is None: psosearch.setIndividualWeight(individualWeight)
    if not inertiaWeight is None: psosearch.setInertiaWeight(inertiaWeight)
    if not iterations is None: psosearch.setIterations(iterations)
    if not logFile is None: psosearch.setLogFile(logFile)
    if not mutationProb is None: psosearch.setMutationProb(mutationProb)
    if not mutationType is None: psosearch.setMutationType(mutationType)
    if not populationSize is None: psosearch.setPopulationSize(populationSize)
    if not reportFrequency is None: psosearch.setReportFrequency(reportFrequency)
    if not seed is None: psosearch.setSeed(seed)
    if not socialWeight is None: psosearch.setSocialWeight(socialWeight)
    if not startSet is None: psosearch.setStartSet(startSet)
    return psosearch

def WASTabuSearch(diversificationProb=1.0,
                  initialSize=-1,
                  numNeighborhood=-1,
                  seed=1):
    tabusearch = TabuSearch()
    if not diversificationProb is None: tabusearch.setDiversificationProb(diversificationProb)
    if not initialSize is None: tabusearch.setInitialSize(initialSize)
    if not numNeighborhood is None: tabusearch.setNumNeighborhood(numNeighborhood)
    if not seed is None: tabusearch.setSeed(seed)
    return tabusearch

def WASSubsetSizeForwardSelection(lookupCacheSize=1,
                                  numSubsetSizeCVFolds=5,
                                  numUsedAttributes=50,
                                  performRanking=True,
                                  seed=1,
                                  subsetSizeEvaluator=None,
                                  type=None,
                                  verbose=False):
    subsetsizeforwardselection = SubsetSizeForwardSelection()
    if not lookupCacheSize is None: subsetsizeforwardselection.setLookupCacheSize(lookupCacheSize)
    if not numSubsetSizeCVFolds is None: subsetsizeforwardselection.setNumSubsetSizeCVFolds(numSubsetSizeCVFolds)
    if not numUsedAttributes is None: subsetsizeforwardselection.setNumUsedAttributes(numUsedAttributes)
    if not performRanking is None: subsetsizeforwardselection.setPerformRanking(performRanking)
    if not seed is None: subsetsizeforwardselection.setSeed(seed)
    if not subsetSizeEvaluator is None: subsetsizeforwardselection.setSubsetSizeEvaluator(subsetSizeEvaluator)
    if not type is None: subsetsizeforwardselection.setType(type)
    if not verbose is None: subsetsizeforwardselection.setVerbose(verbose)
    return subsetsizeforwardselection

def WASExhaustiveSearch(verbose=False):
    exhaustivesearch = ExhaustiveSearch()
    if not verbose is None: exhaustivesearch.setVerbose(verbose)
    return exhaustivesearch

def WASRandomSearch(searchPercent=25.0,
                    startSet='',
                    verbose=False):
    randomsearch = RandomSearch()
    if not searchPercent is None: randomsearch.setSearchPercent(searchPercent)
    if not startSet is None: randomsearch.setStartSet(startSet)
    if not verbose is None: randomsearch.setVerbose(verbose)
    return randomsearch

def WASRanker(generateRanking=True,
              numToSelect=-1,
              startSet='',
              threshold=-1.79769313486e+308):
    ranker = Ranker()
    if not generateRanking is None: ranker.setGenerateRanking(generateRanking)
    if not numToSelect is None: ranker.setNumToSelect(numToSelect)
    if not startSet is None: ranker.setStartSet(startSet)
    if not threshold is None: ranker.setThreshold(threshold)
    return ranker

def WASFCBFSearch(generateDataOutput=False,
                  generateRanking=True,
                  numToSelect=-1,
                  startSet='',
                  threshold=-1.79769313486e+308):
    fcbfsearch = FCBFSearch()
    if not generateDataOutput is None: fcbfsearch.setGenerateDataOutput(generateDataOutput)
    if not generateRanking is None: fcbfsearch.setGenerateRanking(generateRanking)
    if not numToSelect is None: fcbfsearch.setNumToSelect(numToSelect)
    if not startSet is None: fcbfsearch.setStartSet(startSet)
    if not threshold is None: fcbfsearch.setThreshold(threshold)
    return fcbfsearch

def WASRaceSearch(attributeEvaluator=None,
                  debug=False,
                  foldsType=None,
                  generateRanking=False,
                  numToSelect=-1,
                  raceType=None,
                  selectionThreshold=-1.79769313486e+308,
                  significanceLevel=0.001,
                  threshold=0.001):
    racesearch = RaceSearch()
    if not attributeEvaluator is None: racesearch.setAttributeEvaluator(attributeEvaluator)
    if not debug is None: racesearch.setDebug(debug)
    if not foldsType is None: racesearch.setFoldsType(foldsType)
    if not generateRanking is None: racesearch.setGenerateRanking(generateRanking)
    if not numToSelect is None: racesearch.setNumToSelect(numToSelect)
    if not raceType is None: racesearch.setRaceType(raceType)
    if not selectionThreshold is None: racesearch.setSelectionThreshold(selectionThreshold)
    if not significanceLevel is None: racesearch.setSignificanceLevel(significanceLevel)
    if not threshold is None: racesearch.setThreshold(threshold)
    return racesearch

def WASScatterSearchV1(combination=None,
                       debug=True,
                       populationSize=-1,
                       seed=1,
                       threshold=0.0):
    scattersearchv1 = ScatterSearchV1()
    if not combination is None: scattersearchv1.setCombination(combination)
    if not debug is None: scattersearchv1.setDebug(debug)
    if not populationSize is None: scattersearchv1.setPopulationSize(populationSize)
    if not seed is None: scattersearchv1.setSeed(seed)
    if not threshold is None: scattersearchv1.setThreshold(threshold)
    return scattersearchv1

def WASGreedyStepwise(conservativeForwardSelection=False,
                      generateRanking=False,
                      numToSelect=-1,
                      searchBackwards=False,
                      startSet='',
                      threshold=-1.79769313486e+308):
    greedystepwise = GreedyStepwise()
    if not conservativeForwardSelection is None: greedystepwise.setConservativeForwardSelection(conservativeForwardSelection)
    if not generateRanking is None: greedystepwise.setGenerateRanking(generateRanking)
    if not numToSelect is None: greedystepwise.setNumToSelect(numToSelect)
    if not searchBackwards is None: greedystepwise.setSearchBackwards(searchBackwards)
    if not startSet is None: greedystepwise.setStartSet(startSet)
    if not threshold is None: greedystepwise.setThreshold(threshold)
    return greedystepwise