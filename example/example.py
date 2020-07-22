from isodisreg import idr
import numpy as np

# Get data
rain = idr.load_rain()
varNames = rain.columns[3:55]
X = rain[varNames][0:185]
y = rain['obs'][0:185]

# Define groups and orders
values = np.ones(52)+1
values[0:2] = 1
groups = dict(zip(varNames, values))
orders = dict({"1":"comp", "2":"icx"})

# compute idr
fit = idr.idr(y = y, X = X, orders = orders, groups = groups)

# fit idr / make prediction
data = rain[varNames][185:190]

preds1 = fit.predict()
preds2 = fit.predict(data)


# evaluation

obs = rain['obs'][185:186]

crps_values = preds2.crps(obs)

eval_cdf = preds1.cdf(thresholds = 0)

preds1.plot()

