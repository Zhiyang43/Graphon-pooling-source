import os
import pickle
import numpy as np

thisFilename = 'sourceLocGNN-graphon-20211215160232'
saveDirRoot = 'experiments' # In this case, relative location
saveDir = os.path.join(saveDirRoot, thisFilename)
saveDirVars = os.path.join(saveDir, 'trainVars')
pathToFile = os.path.join(saveDirVars,'trainVarsG01.pkl')
# load : get the data from file
data = pickle.load(open(pathToFile, "rb"))
# loads : get the data from var
# print(np.mean(data['timeTrain']['GraphonGNNRegularIntegrationG01']), np.std(data['timeTrain']['GraphonGNNRegularIntegrationG01']))
print(np.mean(data['timeTrain']['GraphonGNNRegularSamplingG01']), np.std(data['timeTrain']['GraphonGNNRegularSamplingG01']))
# print(np.mean(data['timeTrain']['GraphonGNNIrregularIntegrationG01']), np.std(data['timeTrain']['GraphonGNNIrregularIntegrationG01']))
# print(np.mean(data['timeTrain']['GraphonGNNIrregularSamplingG01']), np.std(data['timeTrain']['GraphonGNNIrregularSamplingG01']))
print(np.mean(data['timeTrain']['CoarseningG01']), np.std(data['timeTrain']['CoarseningG01']))

# print('Graphon %f +- %f' %(np.mean(data['timeTrain']) ,np.std(data['timeTrain'])))
# pathToFile = os.path.join(saveDirVars,'SelGNNdegR00trainVars.pkl')
# data = pickle.load(open(pathToFile, "rb"))
# # loads : get the data from var
# print('SelGNNdeg %f +- %f' %(np.mean(data['timeTrain']), np.std(data['timeTrain'])))
# print(data['timeTrain'])

