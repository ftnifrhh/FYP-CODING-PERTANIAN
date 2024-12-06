# fyp progress : multiple linear regression ( hasil pertanian )

import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.diagnostic import het_breuschpagan
from scipy.stats import shapiro,boxcox
import scipy.stats as stats

sns.set()
from sklearn.linear_model import LinearRegression

# STEP 1 : data

print(" STEP 1 : LOAD DATA ")
#  -------- load data  ----------------
raw_data_path = r"C:\Users\Fatini\OneDrive - Universiti Malaya\DATA FOR FYP.xlsx"
raw_data = pd.read_excel(raw_data_path, sheet_name='HASIL PERTANIAN ( full )')
raw_data = raw_data.apply(pd.to_numeric, errors='coerce').fillna(0).astype(int)

#---------detect any missing value----------
print("Detecting any missing value")
print( raw_data.isnull().sum() )
print("                                                                                    ")

# 
# STEP 2 : EDA
print("                                                                     ")
print(" STEP 2 : EDA  ")
# ---------show the discriptive statistic from the data----------
pd.set_option('display.max_columns', None)
raw_data = raw_data.drop(columns=['YEAR'])
print(raw_data.describe(include='all'))
median_box=pd.DataFrame(raw_data.median(),columns=['Median'])
print(median_box)
# #

# ------------------------------histogram----------------------------- ( done and perfect )
plt.figure()
plt.hist(raw_data['job vacancies PERTANIAN'],color='lightblue',edgecolor='black')
plt.xlabel('JOB VACANCIES (PERTANIAN)')
plt.ylabel('Frequency')
plt.title('Job Vacancies (Pertanian)')
median_jobdemands=np.median(raw_data['job vacancies PERTANIAN'])
mean_jobdemands=np.mean(raw_data['job vacancies PERTANIAN'])
plt.axvline(median_jobdemands,color='red',linestyle='dashed', linewidth=2, label=f'Median: {median_jobdemands}')
plt.axvline(mean_jobdemands, color='green', linestyle='dashed', linewidth=2, label=f'Mean: {mean_jobdemands}')
plt.legend()
plt.grid(True)

plt.figure()
plt.hist(raw_data['GETAH ASLI (NATURAL RUBBER)'],color='lightblue',edgecolor='black')
plt.xlabel('GETAH ASLI (NATURAL RUBBER)')
plt.ylabel('Frequency')
plt.title('GETAH ASLI (NATURAL RUBBER)')
median_jobdemands=np.median(raw_data['GETAH ASLI (NATURAL RUBBER)'])
mean_jobdemands=np.mean(raw_data['GETAH ASLI (NATURAL RUBBER)'])
plt.axvline(median_jobdemands,color='red',linestyle='dashed', linewidth=2, label=f'Median: {median_jobdemands}')
plt.axvline(mean_jobdemands, color='green', linestyle='dashed', linewidth=2, label=f'Mean: {mean_jobdemands}')
plt.legend()
plt.grid(True)

plt.figure()
plt.hist(raw_data['BIJI KOKO'],color='lightblue',edgecolor='black')
plt.xlabel('BIJI KOKO')
plt.ylabel('Frequency')
plt.title('BIJI KOKO')
median_jobdemands=np.median(raw_data['BIJI KOKO'])
mean_jobdemands=np.mean(raw_data['BIJI KOKO'])
plt.axvline(median_jobdemands,color='red',linestyle='dashed', linewidth=2, label=f'Median: {median_jobdemands}')
plt.axvline(mean_jobdemands, color='green', linestyle='dashed', linewidth=2, label=f'Mean: {mean_jobdemands}')
plt.legend()
plt.grid(True)

plt.figure()
plt.hist(raw_data['NANAS'],color='lightblue',edgecolor='black')
plt.xlabel('NANAS')
plt.ylabel('Frequency')
plt.title('NANAS')
median_jobdemands=np.median(raw_data['NANAS'])
mean_jobdemands=np.mean(raw_data['NANAS'])
plt.axvline(median_jobdemands,color='red',linestyle='dashed', linewidth=2, label=f'Median: {median_jobdemands}')
plt.axvline(mean_jobdemands, color='green', linestyle='dashed', linewidth=2, label=f'Mean: {mean_jobdemands}')
plt.legend()
plt.grid(True)

plt.show()
# ----------------------------------------------boxplots-----------------------------------------------------------(done)
import numpy as np
import matplotlib.pyplot as plt


m=raw_data['job vacancies PERTANIAN']
q1 = np.percentile(m, 25)
q3 = np.percentile(m, 75)
iqr = q3 - q1
median=np.median(m)
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr
outliers = m[(m< lower_bound) | (m > upper_bound)]
print("                                                                     ")
print(" OUTLIERS FOR JOB VACANCIES PERTANIAN  ")
print(outliers)
whisker_lower = np.min(m[m >= lower_bound])
whisker_upper = np.max(m[m <= upper_bound])
plt.figure(figsize=(10, 6))
plt.boxplot(m, vert=False, patch_artist=True, boxprops={'facecolor':'lightblue'}, flierprops={'marker': 'o', 'markerfacecolor':'red', 'markeredgecolor':'red'})
plt.xlabel('Values')
print("                                                                     ")
plt.title('job vacancies PERTANIAN')
plt.grid(True)
for outlier in outliers:
    plt.text(outlier, 1.05, f'{outlier}', color='red', ha='center', va='bottom')  # Adjusted y-position
plt.text(q1, 1.1, f'Q1: {q1}', color='blue', ha='center')
plt.text(q3, 1.1, f'Q3: {q3}', color='blue', ha='center')
plt.text(median, 1.15, f'Median: {median}', color='green', ha='center')
plt.text(whisker_lower, 1.05, f'Whisker Lower: {whisker_lower}', color='purple', ha='center')
plt.text(whisker_upper, 1.05, f'Whisker Upper: {whisker_upper}', color='purple', ha='center')
plt.show()

k=raw_data['GETAH ASLI (NATURAL RUBBER)']
q1 = np.percentile(k, 25)
q3 = np.percentile(k, 75)
iqr = q3 - q1
median=np.median(k)
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr
outliers = k[(k< lower_bound) | (k > upper_bound)]
print(" OUTLIERS FOR GETAH ASLI (NATURAL RUBBER) ")
print(outliers)
whisker_lower = np.min(k[k >= lower_bound])
whisker_upper = np.max(k[k <= upper_bound])
plt.figure(figsize=(10, 6))
plt.boxplot(k, vert=False, patch_artist=True, boxprops={'facecolor':'lightblue'}, flierprops={'marker': 'o', 'markerfacecolor':'red', 'markeredgecolor':'red'})
plt.xlabel('Values')
print("                                                                     ")
plt.title('GETAH ASLI (NATURAL RUBBER)')
plt.grid(True)
for outlier in outliers:
    plt.text(outlier, 1.05, f'{outlier}', color='red', ha='center', va='bottom')  # Adjusted y-position
plt.text(q1, 1.1, f'Q1: {q1}', color='blue', ha='center')
plt.text(q3, 1.1, f'Q3: {q3}', color='blue', ha='center')
plt.text(median, 1.15, f'Median: {median}', color='green', ha='center')
plt.text(whisker_lower, 1.05, f'Whisker Lower: {whisker_lower}', color='purple', ha='center')
plt.text(whisker_upper, 1.05, f'Whisker Upper: {whisker_upper}', color='purple', ha='center')
plt.show()

n=raw_data['BIJI KOKO']
q1 = np.percentile(n, 25)
q3 = np.percentile(n, 75)
iqr = q3 - q1
median=np.median(n)
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr
outliers = n[(n< lower_bound) | (n > upper_bound)]
print("                                                                     ")
print(" OUTLIERS FOR BIJI KOKO  ")
print(outliers)
whisker_lower = np.min(n[n >= lower_bound])
whisker_upper = np.max(n[n <= upper_bound])
plt.figure(figsize=(10, 6))
plt.boxplot(n, vert=False, patch_artist=True, boxprops={'facecolor':'lightblue'}, flierprops={'marker': 'o', 'markerfacecolor':'red', 'markeredgecolor':'red'})
plt.xlabel('Values')
print("                                                                     ")
plt.title('BIJI KOKO')
plt.grid(True)
for outlier in outliers:
    plt.text(outlier, 1.05, f'{outlier}', color='red', ha='center', va='bottom')  # Adjusted y-position
plt.text(q1, 1.1, f'Q1: {q1}', color='blue', ha='center')
plt.text(q3, 1.1, f'Q3: {q3}', color='blue', ha='center')
plt.text(median, 1.15, f'Median: {median}', color='green', ha='center')
plt.text(whisker_lower, 1.05, f'Whisker Lower: {whisker_lower}', color='purple', ha='center')
plt.text(whisker_upper, 1.05, f'Whisker Upper: {whisker_upper}', color='purple', ha='center')
plt.show()
 
w=raw_data['NANAS']
q1 = np.percentile(w, 25)
q3 = np.percentile(w, 75)
iqr = q3 - q1
median=np.median(w)
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr
outliers = w[(w< lower_bound) | (w > upper_bound)]
print(" OUTLIERS FOR NANAS ")
print(outliers)
whisker_lower = np.min(w[w >= lower_bound])
whisker_upper = np.max(w[w <= upper_bound])
plt.figure(figsize=(10, 6))
plt.boxplot(w, vert=False, patch_artist=True, boxprops={'facecolor':'lightblue'}, flierprops={'marker': 'o', 'markerfacecolor':'red', 'markeredgecolor':'red'})
plt.xlabel('Values')
print("                                                                     ")
plt.title('NANAS')
plt.grid(True)
for outlier in outliers:
    plt.text(outlier, 1.05, f'{outlier}', color='red', ha='center', va='bottom')  # Adjusted y-position
plt.text(q1, 1.1, f'Q1: {q1}', color='blue', ha='center')
plt.text(q3, 1.1, f'Q3: {q3}', color='blue', ha='center')
plt.text(median, 1.15, f'Median: {median}', color='green', ha='center')
plt.text(whisker_lower, 1.05, f'Whisker Lower: {whisker_lower}', color='purple', ha='center')
plt.text(whisker_upper, 1.05, f'Whisker Upper: {whisker_upper}', color='purple', ha='center')
plt.show()


# STEP 3 : relax all the OLS assumption
print("                                                                     ")
print(" STEP 3 : CHECKING FOR OLS ASSUMPTION ")
# 1 : check the linearity (DONE)
# Log transformation (ensure values are positive)

log_pertanian =raw_data['job vacancies PERTANIAN'] 

print("                                                                     ")
print(" 1 : CHECK FOR LINEARITY FOR EACH INDEPENDENT VARIABLE ")
print('********************************************************************')
print(' GETAH ASLI ')
x=raw_data['GETAH ASLI (NATURAL RUBBER)']
y=log_pertanian #log format
# 
plt.scatter(x, y)
plt.title('GETAH ASLI (NATURAL RUBBER)')
x_constant=sm.add_constant(x.dropna())
result=sm.OLS(y,x_constant).fit()
intercept,slope=result.params
y_pred=slope*x+intercept # y prediction is in log format # is the same as y_pred_log = result.predict(x_constant)
residuals_log = result.resid.reset_index(drop=True) #residual from the regression model in log format
fitted_values=result.fittedvalues

#---------- test for heteroscedacity : Breusch-Pagan Test
bp_test = het_breuschpagan(residuals_log, x_constant)
lm_stat, lm_pvalue, f_stat, f_pvalue = bp_test

print(f'LM Statistic: {lm_stat}')
print(f'LM p-value: {lm_pvalue}')
print(f'F Statistic: {f_stat}')
print(f'F p-value: {f_pvalue}')


alpha = 0.05  # significance level
if lm_pvalue < alpha:
    print("Reject the null hypothesis: Heteroscedasticity is present.")
else:
    print("Fail to reject the null hypothesis: No evidence of heteroscedasticity.")
# # 
# 
# #----------performance of our regression
# print("Performance Of The Regression for Getah Asli") 
# real=y.reset_index(drop=True) # real from the data
# newy_pred=y_pred.reset_index(drop=True)
# target=newy_pred
# residual=np.absolute (real-target)
# differences=np.absolute ( residual/ real * 100 )
# df_getah=pd.DataFrame( {'Real': real , 'target': target , 'residual':residual,'differences':differences})
# print(df_getah)

#---------to see the trend in our linear regression
plt.plot(x,y_pred, lw=4, c='orange', label='regression line')
plt.show()
# 
print('******************************************************************** ')
print('BIJI KOKO')
x=raw_data['BIJI KOKO']
y=log_pertanian #log format
plt.scatter(x, y)
plt.title('BIJI KOKO')
x_constant=sm.add_constant(x.dropna())
result=sm.OLS(y,x_constant).fit()
intercept,slope=result.params
y_pred1=slope*x+intercept #log format # is the same as y_pred_log = result.predict(x_constant)
residuals_log = result.resid.reset_index(drop=True) #residual from the regression model #log format
fitted_values=result.fittedvalues

#---------- test for heteroscedacity : Breusch-Pagan Test
bp_test = het_breuschpagan(residuals_log, x_constant)
lm_stat, lm_pvalue, f_stat, f_pvalue = bp_test

print(f'LM Statistic: {lm_stat}')
print(f'LM p-value: {lm_pvalue}')
print(f'F Statistic: {f_stat}')
print(f'F p-value: {f_pvalue}')


alpha = 0.05  # significance level
if lm_pvalue < alpha:
    print("Reject the null hypothesis: Heteroscedasticity is present.")
else:
    print("Fail to reject the null hypothesis: No evidence of heteroscedasticity.")

# 
#----------performance of our regression
# print("performance of the regression for Biji Koko ")
# real=y.reset_index(drop=True)# real from the data
# newy_pred1=y_pred1.reset_index(drop=True)
# target=newy_pred1
# residual=np.absolute (real-target)
# differences=np.absolute ( residual/ real * 100 )
# df_getah=pd.DataFrame( {'Real': real , 'target': target , 'residual':residual,'differences':differences})
# print(df_getah)

#---------to see the trend in our linear regression
plt.plot(x,y_pred1, lw=4, c='orange', label='regression line')
plt.show()
#

print('******************************************************************** ')
print(' NANAS ')
x=raw_data['NANAS']
y=log_pertanian #log format
# y=raw_data['job vacancies PERTANIAN']
plt.scatter(x, y)
plt.title('NANAS')
x_constant=sm.add_constant(x.dropna())
result=sm.OLS(y,x_constant).fit()
intercept,slope=result.params
y_pred2=slope*x+intercept #log format # is the same as y_pred_log = result.predict(x_constant)
residuals_log = result.resid.reset_index(drop=True) #residual from the regression model #log format
fitted_values=result.fittedvalues
# 
#---------- test for heteroscedacity : Breusch-Pagan Test
bp_test = het_breuschpagan(residuals_log, x_constant)
lm_stat, lm_pvalue, f_stat, f_pvalue = bp_test

print(f'LM Statistic: {lm_stat}')
print(f'LM p-value: {lm_pvalue}')
print(f'F Statistic: {f_stat}')
print(f'F p-value: {f_pvalue}')


alpha = 0.05  # significance level
if lm_pvalue < alpha:
    print("Reject the null hypothesis: Heteroscedasticity is present.")
else:
    print("Fail to reject the null hypothesis: No evidence of heteroscedasticity.")


#----------performance of our regression
# print(' performance of the regression for NANAS')
# real=y.reset_index(drop=True)# real from the data
# newy_pred2=y_pred2.reset_index(drop=True)
# target=newy_pred2
# residual=np.absolute (real-target)
# differences=np.absolute ( residual/ real * 100 )
# df_getah=pd.DataFrame( {'Real': real , 'target': target , 'residual':residual,'differences':differences})
# print(df_getah)

#---------to see the trend in our linear regression
plt.plot(x,y_pred2, lw=4, c='orange', label='regression line')
plt.show()
# 
# 2 : no endogeneity of regressor (checking for the correlation between error and independent variable) (below) (DONE)
# 3 : normality and homoscedacity
# -- we want to see whether the error(residual) is normal or not (below) (DONE)
# -- error mean is 0 : we can have intercept ( OLS ASSUMPTION) (DONE)
print("                                                                     ")
print(" 3 : CHECKING IF THE ERROR MEAN IS 0 BY HAVING CONSTANT IN OUR REGRESSION MODEL ")
# --homoscedacity : just look at the scatter plot trend (Breusch Pagan Test) (DONE)
# 4 : no autocorrelation between the errors : look at Duibin-Watson (OLS ASSUMPTION) (DONE)
print("                                                                     ")
print(" 4 : CHECK FOR AUTOCORRELATION BETWEEN ERRORS THROUGH DUIBIN WATSON ")
# 5 : no multicollinearity between the independent variable (DONE)
print("                                                                     ")
print(" 5 : CHECK FOR MULTICOLLINEARITY BETWEEN INDEPENDENT VARIABLES BY VIF AND HEATMAP ")
from statsmodels.stats.outliers_influence  import variance_inflation_factor
variables = raw_data[ [ 'GETAH ASLI (NATURAL RUBBER)', 'BIJI KOKO', 'NANAS']]
vif=pd.DataFrame()
vif["VIF"]=[variance_inflation_factor ( variables.values,i) for i in range(variables.shape[1])]
vif["features"]=variables.columns
print(vif)

correlation_matrix = variables.corr()

plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5, fmt=".2f")
plt.title('Correlation Heatmap of Independent Variables')
plt.show()


# # # # STEP 4 : regression model ( with split test data )
# # # #----------------------------------------------------------------------------------------------------------------------
# 
#   declare inputs and targets

inputs = raw_data[['GETAH ASLI (NATURAL RUBBER)', 'BIJI KOKO', 'NANAS']]
targets= log_pertanian
#   train test split (we are just splitting the data into train data and test data)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(inputs,targets, test_size=0.3, random_state=365)
#   we are creating the regression
reg=LinearRegression()
reg.fit(x_train,y_train)
# Now use statsmodels to get detailed output this onE , untuk elak overfitting issue : but this one just by the training data only
X_with_constant = sm.add_constant(x_train)
ols_model = sm.OLS(y_train, X_with_constant)
results = ols_model.fit()
fitted_values = results.fittedvalues
residuals=results.resid
print(results.summary())

# # # #------------------------------------------------------------------------------------------------------------------------
# # Train the model (training data)

# # Make predictions on training data
y_real= reg.predict(x_train)

# OLS ASSUMPTION NO 3 ( BREUSCH PAGAN TEST AND RESIDUALS VS FITTED PLOT )
print("                                                                     ")
print(" 3 : CHECK FOR HOMOSCEDACITY BY BREUSHPAGAN TEST AND  RESIDUALS VS FITTED VALUES ")
#---------- test for heteroscedacity : Breusch-Pagan Test
bp_test = het_breuschpagan(residuals, X_with_constant)
lm_stat, lm_pvalue, f_stat, f_pvalue = bp_test

print(f'LM Statistic: {lm_stat}')
print(f'LM p-value: {lm_pvalue}')
print(f'F Statistic: {f_stat}')
print(f'F p-value: {f_pvalue}')


alpha = 0.05  # significance level
if lm_pvalue < alpha:
    print("Reject the null hypothesis: Heteroscedasticity is present.")
else:
    print("Fail to reject the null hypothesis: No evidence of heteroscedasticity.")
    
# we want to see the difference between fitted and residuals
plt.scatter(fitted_values, residuals) #plt.scatter(x,y)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel("Fitted Values")
plt.ylabel("Residuals")
plt.title("Residuals vs Fitted Values")
plt.show()
print('                                                         ')

# OLS ASSUMPRION NO 3 : to see the pdf of residual use qq-plot : to see if the error follow normal distribution
print("                                                                     ")
print(" 3 : CHECK FOR NORMALITY OF RESIDUAL ERROR ")
sm.qqplot(residuals, line='s')
plt.title("Q-Q Plot of Model Residuals")
plt.show()

shapiro_test=shapiro(residuals)
shapiro_statistic = shapiro_test.statistic
shapiro_p_value = shapiro_test.pvalue

print(f"Shapiro-Wilk Test Statistic: {shapiro_statistic}")
print(f"Shapiro-Wilk Test p-value: {shapiro_p_value}")
alpha = 0.05
if shapiro_p_value > alpha:
    print("Residuals are likely normally distributed (fail to reject H0).")
else:
    print("Residuals are not normally distributed (reject H0).")
 
# OLS ASSUMPTION NO 2
print("                                                                     ")
print(" 2 : CHECK FOR NO ENDEGENOITY OF REGRESSOR BY PEARSONS'S CORRELATION ")
for column in inputs.columns:
    correlation = residuals.corr(inputs[column])
    print(f'Correlation of residuals (error) with {column}: {correlation}')



print(" ================================================================================================")
print(" PERFORMANCE OF OUR REGRESSION MODEL  ")
print("                                          ")

df_pf=pd.DataFrame( y_real , columns=['Real'] )
y_train = pd.Series(y_train)
df_pf['Target'] = y_train.reset_index(drop=True)# if index diorg mcm tak betul, susun balik
df_pf['Residual']=np.absolute (df_pf['Real']-df_pf['Target'])
df_pf['Diferrences %' ] = np.absolute ( df_pf['Residual'] / df_pf ['Target'] * 100 )
pd.set_option('display.float_format', lambda x : '%.2f' % x)
df_pf.sort_values(by=['Diferrences %' ] ) # we  may see the differences in percent 
print(df_pf.describe())
print("                                                                     ")
print(df_pf)

#------------------------------------------------------------------------------------------------------------
# # # Test the model ( testing data )
# y_real2= reg.predict(x_test)
# # 
# # we want to see the difference between y test and y real
# plt.scatter(y_test, y_real2) #plt.scatter(x,y)
# plt.xlabel('Predicted y (y_test)', size=18)
# plt.ylabel('Real y (y_real)', size=18)
# # plt.xlim(min(y_train)+40000, max(y_train)+40000)
# # plt.ylim(min(y_real)+40000, max(y_real)+40000)
# plt.title(' y real vs y test on Test Data')
# plt.show()
# 
# #performance
# 
# df_pf=pd.DataFrame( y_real2 , columns=['Real'] )
# y_test = pd.Series(y_test)
# df_pf['Target'] = y_test.reset_index(drop=True) # if index diorg mcm tak betul, susun balik
# df_pf['Residual']=np.absolute (df_pf['Real']-df_pf['Target'])
# df_pf['Diferrences %' ] = np.absolute ( df_pf['Residual'] / df_pf ['Target'] * 100 )
# pd.set_option('display.float_format', lambda x : '%.2f' % x)
# df_pf.sort_values(by=['Diferrences %' ] ) # we  may see the differences in percent 
# print(df_pf.describe())
# print(df_pf)

