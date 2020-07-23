from isodisreg import idr
import numpy as np
import pandas as pd
import matplotlib as plt

# Get data
rain = idr.load_rain()
varNames = rain.columns[3:55]

###############################################
# 1. Exampe for functions idr() and predict() #
###############################################

X = rain[varNames][0:185]
y = rain['obs'][0:185]
data = rain[varNames][185:186]

# Define groups and orders
values = np.ones(52)+1
values[0:2] = 1
groups = dict(zip(varNames, values))
orders = dict({"1":"comp", "2":"icx"})

# compute idr
fit = idr.idr(y = y, X = X, orders = orders, groups = groups)

# fit idr / make prediction
preds1 = fit.predict()
preds2 = fit.predict(data)

#########################################################################
# 2. Exampe for functions cdf(), qpred(), qscore(), bscore() and crps() #
#########################################################################

# cdf
X = rain[["HRES"]][0:3*365]
y = rain['obs'][0:3*365]
data = pd.DataFrame({"HRES": [0, 0.5, 1]}, columns = ["HRES"])
fit = idr.idr(y = y, X = X)
preds1 = fit.predict(data)
cdf0 = preds1.cdf(thresholds = 0)
print(1-np.array(cdf0))

# qpred
data = pd.DataFrame({"HRES": [2.5, 5, 10]}, columns = ["HRES"])
preds2 = fit.predict(data)
qpedict = preds2.qpred(quantiles = 0.95)

# qscore
data = rain[["HRES"]][3*365:5*365]
obs = rain["obs"][3*365:5*365]
preds3 = fit.predict(data)
idrMAE = np.mean(preds3.qscore(0.5, obs))
print("Mean Qscore:", idrMAE)

# bscore
idrBscore = np.mean(preds3.bscore(thresholds = 0, y = obs))
print("Mean Bscore:", idrBscore)

# crps
idrCRPS = np.mean(preds3.crps(obs))
print("Mean CRPS:",idrCRPS)


#################################
# 3. Exampe for functions pit() #
#################################

X = rain[["HRES"]][0:4*365]
y = rain["obs"][0:4*365]
fit = idr.idr(y = y, X = X)
data = rain[["HRES"]][4*365:8*365]
obs = rain["obs"][4*365:8*365]
preds = fit.predict(data = data)
idrPIT = preds.pit(y = obs, seed = 123)
a, b, x = plt.hist(idrPIT, density = True)
plt.title("Postprocessed HRES")
plt.xlabel("Probability Integral Transform")
plt.ylabel("Density")
plt.show()

##################################
# 3. Exampe for functions plot() #
##################################

X = rain[["HRES"]][0:2*365]
y = rain["obs"][0:2*365]
fit = idr.idr(y = y, X = X)
data = pd.DataFrame({"HRES": [0], "CTR": [0]}, columns = ["HRES", "CTR"])
preds = fit.predict(data = data)
preds.plot() 

