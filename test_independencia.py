import pandas as pd
import numpy as np
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from scipy.stats.mstats import kruskal

from statsmodels.sandbox.stats.multicomp import MultiComparison


def independence_test(df_test, column_value="speed", column_factor="city_id", alpha=0.05, decimals=6):
    """
    Function that carries out indepence test given a numerical column and a factor column.
    First, it is checked that assumptions for ANOVA test are fullfilled.
    Second, if so, ANOVA test is carried out. If not, a non parametric test, Kruskal wallis is performed
    Args: A dataframe, a numerical column, a factor column, an alpha factor for the alpha level of the test,
    decimals to round the p-value

    Returns: A dataframe with the results of the test

    """

    ##We are extracting all posible combinations:
    n_cases = list(df_test[column_factor].unique())

    tuple_combinations = [(x, y) for x in n_cases for y in n_cases if n_cases.index(y) > n_cases.index(x)]

    ##Shapiro test to check the normal distribution of residuals
    ##Null hypothesis: data is drawn from normal distribution.
    w_saphiro, pvalue_saphiro = stats.shapiro(df_test[column_value].values)
    df_out = pd.DataFrame(tuple_combinations)

    ### Kurskal test is performed in any case, and use where anova is not usable.
    Kruskal_test = [kruskal(df_test[df_test[column_factor] == combination[0]][column_value].values,
                            df_test[df_test[column_factor] == combination[1]][column_value].values) for combination in
                    tuple_combinations]
    Kruskal_pvalue = [round(x.pvalue, decimals) for x in Kruskal_test]
    Kruskal_H = [x.statistic for x in Kruskal_test]

    ## if saphiro test is passed, we continue calculating Anova:

    if pvalue_saphiro > 0.05:
        print("Saphiro test p-value: %s" % round(pvalue_saphiro, decimals))
        print("Null hypothesis cannot be rejected. We asume residuals are normally distributed")
        ##Now let's check every possible combinatio to perform Bartlet test:
        Bartlett_test = [stats.bartlett(df_test[df_test[column_factor] == combination[0]][column_value],
                                        df_test[df_test[column_factor] == combination[1]][column_value]) for combination
                         in tuple_combinations]
        ##Extracting p-values and statistic values:
        Bartlett_pvalue = [x.pvalue for x in Bartlett_test]
        Bartlett_w = [x.statistic for x in Bartlett_test]
        Anova_test = [stats.f_oneway(df_test[df_test[column_factor] == combination[0]][column_value].values,
                                     df_test[df_test[column_factor] == combination[1]][column_value].values) for
                      combination in tuple_combinations]
        Anova_pvalue = [x.pvalue for x in Anova_test]
        Anova_f = [x.statistic for x in Anova_test]

        df_out = pd.concat(
            [df_out, pd.DataFrame(Bartlett_pvalue), pd.DataFrame(Anova_pvalue), pd.DataFrame(Kruskal_pvalue)], axis=1)
        df_out.columns = ["First_value", "Second_value", "Bartlett_pvalue", "Anova_pvalue", "kruskal_pvalue"]

        df_out["Reject Bartlett null hyp"] = np.where(df_out["Bartlett_pvalue"] < alpha, True, False)
        df_out["Reject Anova null hyp"] = np.where(df_out["Anova_pvalue"] < alpha, True, False)
        ## Kruscal_pvalue
        df_out["kruskal_pvalue"] = np.where(df_out["Reject Bartlett null hyp"] == True, df_out["kruskal_pvalue"],
                                            np.NaN)



    else:
        print("Saphiro test p-value: %s" % round(pvalue_saphiro, 4))
        print("Null hypothesis rejected. We cannot asume residuals are normally distributed")
        print("Using kruskal wallis test")

        df_out = pd.concat([df_out, pd.DataFrame(Kruskal_pvalue), pd.DataFrame(Kruskal_H)], axis=1)
        # df_out=pd.concat([df_out,pd.DataFrame([Kruskal_H])],axis=1)

        df_out.columns = ["First_value", "Second_value", "kruskal_p_value", "kruskal_H_value"]
        df_out["Reject Kruskal null hyp"] = np.where(df_out["kruskal_p_value"] < alpha, True, False)

    return df_out
