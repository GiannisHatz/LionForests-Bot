from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
# new import
from sklearn.calibration import CalibratedClassifierCV
import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
from sklearn.cluster import DBSCAN
from collections import Counter
from LionForests import kmedoids
import shap
import random
from LionForests.lionforests_utility import roundup, path_similarity, path_distance


class LionForests:
    """Class for interpreting random forests classifier through following_breacrumbs technique"""

    def __init__(self, model=None, trained=False, utilizer=None, feature_names=None, class_names=None,
                 categorical_features=None):
        """Init function
        Args:
            model: The trained RF model
            utilizer: The preferred scaler
            feature_names: The names of the features from our dataset
            class_names: The names of the classes from our dataset
        Attributes:
            model: The classifier/ regression model
            utilizer: The scaler, if any
            trees: The trees of an trained ensemble system
            feature_names: The names of the features
            class_names: The names of the two classes
            accuracy: The accuracy of the model (accuracy for classification, mse for regression):
            min_max_feature_values: A helping dictionary for the path/feature reduction process
            number_of_estimators: The amount of trees
            ranked_features: The features ranked based on SHAP Values (Small-Medium Datasets) or Feature Importance (Huge Datasets)
        """
        self.model = model
        self.utilizer = utilizer
        self.trees = None
        if model is not None:
            self.trees = model.estimators_
        self.feature_names = feature_names
        self.categorical_features = categorical_features
        self.class_names = class_names
        self.accuracy = 0
        self.min_max_feature_values = {}
        self.number_of_estimators = 0
        self.ranked_features = {}
        self.quorum = 0
        # new
        self.feature_statistics = {}
        self.calibrator = None
        # end new
        if trained:
            self._trained()

    def _trained(self):
        self.trees = self.model.estimators_
        self.number_of_estimators = self.model.n_estimators
        self.quorum = int(self.number_of_estimators / 2 + 1)

    def transform_categorical_data(self, train_data, train_target, feature_names):
        self.onehot_data = to_onehot_data
        self.onehot_transformation = True
        self.numerical_data = [i for i in feature_names if i not in to_onehot_data]

        numerical = pd.DataFrame(train_data, columns=feature_names)[self.numerical_data].values
        num = pd.DataFrame(numerical, columns=self.numerical_data)
        if self.onehot_transformation:
            onehot_encoded = pd.DataFrame(train_data, columns=feature_names)[to_onehot_data].values
            enc = OneHotEncoder(handle_unknown='ignore')
            enc.fit(onehot_encoded)
            self.onehot_dictionary = {}
            for i in range(len(to_onehot_data)):
                self.onehot_dictionary[str('x' + str(i))] = to_onehot_data[i]
            self.onehot_features = enc.get_feature_names()
            one = pd.DataFrame(enc.transform(onehot_encoded).A, columns=self.onehot_features)

        if self.ordinal_transformation:
            ordinal_encoded = pd.DataFrame(train_data, columns=feature_names)[to_ordinal_data].values
            enc = OrdinalEncoder()
            enc.fit(ordinal_encoded)
            self.ordinal_features = to_ordinal_data
            ore = pd.DataFrame(enc.transform(ordinal_encoded), columns=to_ordinal_data)

        if self.onehot_transformation and self.ordinal_transformation:
            all_features = []
            for f in self.onehot_features:
                all_features.append(f)
            for f in self.ordinal_features:
                all_features.append(f)
            for f in self.numerical_data:
                all_features.append(f)
            return pd.concat([one, ore, num], axis=1), train_target, all_features
        elif self.onehot_transformation:
            all_features = []
            for f in self.onehot_features:
                all_features.append(f)
            for f in self.numerical_data:
                all_features.append(f)
            return pd.concat([one, num], axis=1), train_target, all_features
        elif self.ordinal_transformation:
            all_features = []
            for f in self.ordinal_features:
                all_features.append(f)
            for f in self.numerical_data:
                all_features.append(f)
            return pd.concat([ore, num], axis=1), train_target, all_features

    def train(self, train_data, train_target, scaling_method=None, feature_names=None, params=None,
              categorical_features=None):
        """ train function is used to train an RF model and extract information like accuracy, model, trees and
        min_max_feature_values among all trees
        Args:
            train_data: The data we are going to use to train the random forest
            train_target: The targets of our train data
            scaling_method: The preffered scaling method. The deafult is MinMaxScaler with feature range -1 to 1
            feature_names: The names of the features from our dataset
            params: The parameters for our gridSearchCV to select the best RF model
        """
        # new
        self.feature_statistics = self.extract_feature_statistics(train_data)

        self.categorical_features = categorical_features
        if feature_names is not None:
            self.feature_names = feature_names
        if scaling_method is not None:  # load scaling
            self.utilizer = scaling_method
        else:
            self.utilizer = MinMaxScaler(feature_range=(-1, 1))
        self.utilizer.fit(train_data)
        train_data = self.utilizer.transform(train_data)

        random_forest = RandomForestClassifier(random_state=0, n_jobs=-1)
        parameters = params
        if parameters is None:
            parameters = [{
                'max_depth': [5],  # [1, 5, 7, 10],#1, 5, 7, 10
                'max_features': ['sqrt'],  # ['sqrt', 'log2', 0.75, None], #'sqrt', 'log2', 0.75, None
                'bootstrap': [True],  # [True, False], #True, False
                'min_samples_leaf': [2],  # [1, 2, 5, 10, 0.10], #1, 2, 5, 10, 0.10
                'n_estimators': [250]  # [10, 100, 500, 1000] #10, 100, 500, 1000
            }]
        clf = GridSearchCV(estimator=random_forest, param_grid=parameters, cv=10, n_jobs=-1, verbose=1, scoring='f1')
        clf.fit(train_data, train_target)

        self.accuracy = clf.best_score_
        self.model = clf.best_estimator_
        self.trees = self.model.estimators_
        self.number_of_estimators = self.model.n_estimators
        self.quorum = int(self.number_of_estimators / 2 + 1)
        for i in range(len(self.feature_names)):
            self.min_max_feature_values[self.feature_names[i]] = [min(train_data[:, i]), max(train_data[:, i])]
        for ind in range(len(self.class_names)):
            d = {'Feature': feature_names, 'Importance': self.model.feature_importances_}
            self.ranked_features[self.class_names[ind]] = \
                pd.DataFrame(data=d).sort_values(by=['Importance'], ascending=False)['Feature'].values

        # new stuff
        self.calibrator = CalibratedClassifierCV(self.model, cv=2, method='sigmoid')
        self.calibrator.fit(train_data, train_target)
        # end of new stuff

    def path_finder(self, instance, info=False):
        """path_finder function finds
        Args:
            instance: The instance we want to find the paths
            info: If we want to show information about the features in the paths
        Return:
            a list which contains a dictionary with features as keys and their min max ranges as values, as well as the
            number of the paths
        """
        if self.utilizer is not None:
            instance = self.utilizer.transform([instance])[0]
        prediction = int(self.model.predict([instance])[0])
        total_leq = {}  # All the rules with less equal operators ex: a <= 1
        total_b = {}  # All the rules with bigger than operators ex: c > 0.1
        rules = []
        ranges = []
        for tree in self.trees:
            tree_prediction = int(tree.predict([instance])[0])
            if tree_prediction == prediction:
                path = tree.decision_path([instance])
                rule = 'if '
                leq = {}  # leq: less equal ex: x <= 1
                b = {}  # b: bigger ex: x > 0.6
                local_range = {}
                for node in path.indices:
                    feature_id = tree.tree_.feature[node]
                    feature = self.feature_names[feature_id]
                    threshold = tree.tree_.threshold[node]
                    if threshold != -2.0:
                        if instance[feature_id] <= threshold:
                            leq.setdefault(feature, []).append(threshold)
                        else:
                            b.setdefault(feature, []).append(threshold)
                for k in leq:
                    rule = rule + k + "<=" + str(min(leq[k])) + " and "
                    total_leq.setdefault(k, []).append(min(leq[k]))  # !!
                    local_range.setdefault(k, []).append(['<=', min(leq[k])])  # !!
                for k in b:
                    rule = rule + k + ">" + str(max(b[k])) + " and "
                    total_b.setdefault(k, []).append(max(b[k]))  # !!
                    local_range.setdefault(k, []).append(['>', max(b[k])])  # !!
                rule = rule[:-4] + "then " + str(self.class_names[int(tree.predict([instance])[0])])
                rules.append(rule)
                ranges.append(local_range)

        if info:
            print("Number of paths:", len(rules))
            for k in total_leq:
                print(
                    "{:<10} {:<2} {:<7} | {:<7}".format(k[:10], '<=', roundup(min(total_leq[k]), 4), len(total_leq[k])))
            for k in total_b:
                print("{:<10} {:<2} {:<7} | {:<7}".format(k[:10], '>', roundup(max(total_b[k]), 4), len(total_b[k])))
        del instance, prediction, total_leq, total_b
        return [ranges, len(rules)]  # If i want rules at natural language i have to add rules variable here too

    def following_breadcrumbs(self, instance, info=False, reduction=True, save_plots=False, complexity=4,
                              instance_quorum=0, medoids=0):
        """following_breadcrumbs function finds a single range rule which will be the explanation for the prediction
        of this instance
        Args:
            instance: The instance we want to find the paths
            info: If we want to show information about the features in the paths
            reduction: The targets of our train data
            save_plots: The bar and stacked area plots for every feature will be saved
            complexity:
        Return:
            a feature range rule which will be the explanation
        """

        if instance_quorum <= 0:
            instance_quorum = self.quorum

        number_of_medoids = medoids
        if medoids <= 0:
            number_of_medoids = 5
            if self.number_of_estimators < 5:
                number_of_medoids = self.number_of_estimators
            if self.number_of_estimators >= 100:
                number_of_medoids = int(math.ceil(instance_quorum * 3 / 22))  # 1100 = 11 * 100

        rules = self.path_finder(instance, info)[0]
        original_number_of_rules = len(rules)

        items = set()
        for pr in rules:
            for p in pr:
                items.add(p)
        local_feature_names = list(items)
        original_number_of_features = len(local_feature_names)
        if reduction:
            temp_rules = self.reduce_rules(rules, instance_quorum, number_of_medoids)
            if len(temp_rules[0]) != 0:
                rules = temp_rules[0]
                local_feature_names = temp_rules[1]
        rule = "if "
        temp_f_mins = {}
        temp_f_maxs = {}
        feature_rule_limits = {}
        for feature in self.feature_names:
            if feature in local_feature_names:
                bars, bars_len = self._pre_feature_range_caluclation(rules, feature, complexity)
                if bars != False:
                    aggregation = self._aggregated_feature_range(bars, feature, save_plots, complexity)

                    temp_f_mins[feature] = aggregation[0]
                    temp_f_maxs[feature] = aggregation[1]
        f_mins = []
        f_maxs = []
        for feature in self.feature_names:
            if feature in temp_f_mins:
                f_mins.append(temp_f_mins[feature])
            else:
                f_mins.append(0)
            if feature in temp_f_maxs:
                f_maxs.append(temp_f_maxs[feature])
            else:
                f_maxs.append(0)
        if self.utilizer is not None:
            instance = self.utilizer.transform([instance])[0]
        class_name = self.class_names[self.model.predict([instance])[0]]
        if self.categorical_features is not None:
            for ranked_f in self.ranked_features[class_name]:
                f = self.feature_names.index(ranked_f)
                it_was_categ = False
                for ff in self.categorical_features:
                    if ff in self.feature_names[f]:
                        if self.feature_names[f] in local_feature_names:
                            if self.utilizer is not None:
                                mmi = self.utilizer.inverse_transform(np.array([f_mins, f_mins]))[0][f]
                                mma = self.utilizer.inverse_transform(np.array([f_maxs, f_maxs]))[0][f]
                            else:
                                mmi = np.array([f_mins, f_mins])[0][f]
                                mma = np.array([f_maxs, f_maxs])[0][f]

                            if str(round(mma, 3)) == '1.0':
                                feature_rule_limits[self.feature_names[f]] = [mmi, mma]
                                rule = rule + self.feature_names[f] + " & "
                            it_was_categ = True
                if not it_was_categ:
                    if self.feature_names[f] in local_feature_names:
                        if self.utilizer is not None:
                            mmi = self.utilizer.inverse_transform(np.array([f_mins, f_mins]))[0][f]
                            mma = self.utilizer.inverse_transform(np.array([f_maxs, f_maxs]))[0][f]
                        else:
                            mmi = np.array([f_mins, f_mins])[0][f]
                            mma = np.array([f_maxs, f_maxs])[0][f]
                        feature_rule_limits[self.feature_names[f]] = [mmi, mma]
                        rule = rule + str(round(mmi, 3)) + "<=" + self.feature_names[f] + "<=" + str(
                            round(mma, 3)) + " & "
        else:
            for ranked_f in self.ranked_features[class_name]:
                f = self.feature_names.index(ranked_f)
                if self.feature_names[f] in local_feature_names:
                    if self.utilizer is not None:
                        mmi = self.utilizer.inverse_transform(np.array([f_mins, f_mins]))[0][f]
                        mma = self.utilizer.inverse_transform(np.array([f_maxs, f_maxs]))[0][f]
                    else:
                        mmi = np.array([f_mins, f_mins])[0][f]
                        mma = np.array([f_maxs, f_maxs])[0][f]
                    feature_rule_limits[self.feature_names[f]] = [mmi, mma]
                    rule = rule + str(round(mmi, 3)) + "<=" + self.feature_names[f] + "<=" + str(round(mma, 3)) + " & "
        # print(feature_rule_limits) #JM: Added this!
        del temp_f_maxs, temp_f_mins, f_maxs, f_mins  # , feature_rule_limits
        return [rule[:-3] + " then " + class_name, original_number_of_rules, original_number_of_features, len(rules),
                len(local_feature_names)]

    def reduce_rules(self, rules, instance_quorum, number_of_medoids):
        """following_breadcrumbs function finds
        Args:
            instance: The instance we want to find the paths
            reduction: The targets of our train data
            save_plots: The bar and stacked area plots for every feature will be saved
        Return:

        """
        get_itemsets = []
        for pr in rules:
            itemset = []
            for p in pr:
                itemset.append(p)
            get_itemsets.append(itemset)
        te = TransactionEncoder()
        te_ary = te.fit(get_itemsets).transform(get_itemsets)
        df = pd.DataFrame(te_ary, columns=te.columns_)
        frequent_itemsets = association_rules(apriori(df, min_support=0.1, use_colnames=True), metric="support",
                                              min_threshold=0.1).sort_values(by="confidence", ascending=True)
        size = 0
        k = 1
        size_of_ar = len(list(list(frequent_itemsets['antecedents'])))
        items = set()
        reduced_rules = rules
        new_feature_list = []
        for pr in reduced_rules:
            for p in pr:
                items.add(p)
            new_feature_list = list(items)
        while size < instance_quorum and k < size_of_ar:
            feature_set = set()
            for i in range(0, k):
                for j in list(list(frequent_itemsets['antecedents'])[i]):
                    feature_set.add(j)
            new_feature_list = list(feature_set)
            redundant_features = [i for i in self.feature_names if i not in new_feature_list]
            reduced_rules = []
            for i in rules:
                if sum([1 for j in redundant_features if j in i]) == 0:
                    reduced_rules.append(i)
            size = len(reduced_rules)
            k += 1
        del get_itemsets, te, te_ary, df, frequent_itemsets
        if len(reduced_rules) < instance_quorum:
            reduced_rules = rules
            for pr in reduced_rules:
                for p in pr:
                    items.add(p)
                new_feature_list = list(items)
        if len(reduced_rules) > instance_quorum:  # If we need more reduction on path level
            A = []
            for k in range(len(reduced_rules)):
                B = []
                for j in range(len(reduced_rules)):
                    if k == j:
                        B.append(0)  # or 1?
                    else:
                        sim = path_similarity(reduced_rules[k], reduced_rules[j], new_feature_list,
                                              self.min_max_feature_values)
                        # sim = path_distance(reduced_rules[k], reduced_rules[j], new_feature_list,
                        # self.min_max_feature_values) #Tested with distance metric of iForest
                        B.append(1 - sim)
                A.append(B)
            A = np.array(A)
            MS, S = kmedoids.kMedoids(A, number_of_medoids)
            medoids_sorted = sorted(S, key=lambda k: len(S[k]), reverse=True)
            k = 0
            size = 0
            reduced_rules_medoids = []
            while size < instance_quorum and k < len(medoids_sorted):
                for j in S[medoids_sorted[k]]:
                    reduced_rules_medoids.append(reduced_rules[j])
                k += 1
                size = len(reduced_rules_medoids)
            items = set()
            if len(reduced_rules_medoids) >= instance_quorum:
                reduced_rules = reduced_rules_medoids
                for pr in reduced_rules_medoids:
                    for p in pr:
                        items.add(p)
                new_feature_list = list(items)
        if len(reduced_rules) > instance_quorum:
            random.shuffle(reduced_rules)
            reduced_rules = reduced_rules[:instance_quorum]
            items = set()
            for pr in reduced_rules:
                for p in pr:
                    items.add(p)
            new_feature_list = list(items)
        return [reduced_rules, new_feature_list]

    def _pre_feature_range_caluclation(self, rules, feature, complexity=4):
        mi = self.min_max_feature_values[feature][0]
        ma = self.min_max_feature_values[feature][1]
        for i in rules:
            if feature in i:
                if len(i[feature]) == 1:
                    if i[feature][0][0] == "<=":
                        if ma < i[feature][0][1]:
                            ma = i[feature][0][1]
                    else:
                        if mi > i[feature][0][1]:
                            mi = i[feature][0][1]
                else:
                    if mi > i[feature][1][1]:
                        mi = i[feature][1][1]
                    if ma < i[feature][0][1]:
                        ma = i[feature][0][1]

        bars = []
        temp_count = 0
        for i in rules:
            if feature in i:
                temp_count += 1
                if len(i[feature]) == 1:
                    if i[feature][0][0] == "<=":
                        bars.append(np.arange(roundup(mi, complexity), roundup(i[feature][0][1], complexity),
                                              (10 ** (-complexity))))
                    else:
                        bars.append(np.arange(roundup(i[feature][0][1], complexity), roundup(ma, complexity),
                                              (10 ** (-complexity))))
                else:
                    mm = [roundup(i[feature][0][1], complexity), roundup(i[feature][1][1], complexity)]
                    bars.append(np.arange(min(mm), max(mm), (10 ** (-complexity))))
        if temp_count == 0:
            return False, False
        bars_len = [len(bar) for bar in bars]
        return bars, bars_len

    def _aggregated_feature_range(self, bars, feature, save_plots=False, complexity=4):
        """_aggregated_feature_range function returns the min and max value from the intersection of all paths
        Args:
            feature: the feature which range we want to find
            save_plots: if yes then it will save the bar and stacked area plots of each feature
            complexity: determines how many digits we will use to descritize
        Return:
            min max of the intersect area of all paths for a feature
        """
        mi = self.min_max_feature_values[feature][0]
        ma = self.min_max_feature_values[feature][1]

        if save_plots:
            plt.figure(figsize=(16, 10))
            plt.title(feature)
            plt.ylabel('No of Rules')
            plt.xlabel('Value')
            for i in range(len(bars)):
                plt.plot(bars[i], len(bars[i]) * [i + 1], linewidth=8)
            plt.savefig(feature + "BarsPlot.png")

        temp_bars = []
        for i in bars:
            bar = set()
            for j in i:
                bar.add(int((roundup(j, complexity) - roundup(mi, complexity)) * (10 ** complexity)))
            temp_bars.append(bar)
        bars = temp_bars
        del temp_bars

        st = {}
        for i in bars:
            for j in i:
                if not int(j) in st:
                    st[int(j)] = 1
                else:
                    st[int(j)] += 1
        del bars

        x = []
        y = []
        max_v = -1
        for key, value in st.items():
            if max_v < value:
                max_v = value
            x.append((key + int(roundup(mi, complexity) * (10 ** complexity))) / (10 ** complexity))
            y.append(value)
        x, y = zip(*sorted(zip(x, y)))
        x_2 = []
        x_3 = []
        y_2 = []
        for key, value in st.items():
            x_2.append((key + int(roundup(mi, complexity) * (10 ** complexity))) / (10 ** complexity))
            if max_v == value:
                x_3.append((key + int(roundup(mi, complexity) * (10 ** complexity))) / (10 ** complexity))
                y_2.append(value)
            else:
                y_2.append(0)
        del st

        if save_plots:
            x_2, y_2 = zip(*sorted(zip(x_2, y_2)))
            plt.figure(figsize=(16, 10))
            plt.title(feature)
            plt.ylabel('No of Rules')
            plt.xlabel('Value')
            plt.stackplot(x, y)
            plt.stackplot(x, y_2, colors='c')
            plt.savefig(feature + "StackedAreaPlot.png")
        del x, y, x_2, y_2
        return [min(x_3), max(x_3)]

    def calculate_single_feature_values(self, instance, feature, rule, discrete_features):

        """calculate_single_feature_values finds and returns, for a single feature, the following values:
            a) One value just below the lower bound of that feature in the LF rule.
            b) One value just above the higher bound of that feature in the LF rule.
            c) One value (for now, probably gonna be 2 for some features) which belongs in the middle of the previous mentioned bounds.

            This function ignores features that do not have ranges in the rule (transformed categorical).

        Args:
            instance: the instance for which the lf rule was made.
            feature: the feature we want to calculate the values for.
            rule: the lf rule.
            discrete_features: a list containing all the discrete features.
        Return:
            a list of values as [left_value, middle_value, right_value]
        """
        feature_index = self.feature_names.index(feature)
        index_left = self.find_first_digit_index(feature, rule, 'left')
        if index_left == -1:
            # no ranges in the rule, ignore feature.
            return []
        index_right = self.find_first_digit_index(feature, rule, 'right')
        left_value = right_value = middle_value = ""

        while rule[index_left] != " ":
            left_value = left_value + rule[index_left]
            index_left -= 1
        while rule[index_right] != " ":
            right_value = right_value + rule[index_right]
            index_right += 1

        min_value, max_value = self.feature_statistics[feature_index][0], self.feature_statistics[feature_index][1]
        mean_, std_ = self.feature_statistics[feature_index][2], self.feature_statistics[feature_index][3]

        noise = np.random.normal(mean_, std_, 1)[0]  # Gaussian Noise
        # print ('noise is ',noise)

        new_feature_values = []
        if feature in discrete_features:
            new_feature_values = self.calc_single_feature_discrete_values(instance, feature_index, left_value,
                                                                          right_value,
                                                                          noise, min_value, max_value)
        else:
            new_feature_values = self.calc_single_feature_continuous_values(instance, feature_index, left_value,
                                                                            right_value, noise, min_value, max_value)

        return new_feature_values

    def calc_single_feature_discrete_values(self, instance, feature_index, left_value, right_value, noise, min_value,
                                            max_value):
        """calc_single_feature_discrete_values finds and returns, for a single *discrete* feature, the following values:
            a) One value just below the lower bound of that feature in the LF rule.
            b) One value just above the higher bound of that feature in the LF rule.
            c) One value (for now, probably gonna be 2 for some features) which belongs in the middle of the previous mentioned bounds.

            This function ignores features that do not have ranges in the rule (transformed categorical).

        Args:
            instance: the instance for which the lf rule was made.
            feature_index: the index of the feature we need to calculate the values for in the feature names list.
            left_value: the lower bound of the feature in the lf rule.
            right_value: the higher bound of the feature in the lf rule.
            noise: the gaussian noise, calculated using the mean and std of the feature.
            min_value: the minimum accepted value for the feature.
            max_value: the maximum accepted value for the feature.
        Return:
            a list of values as [left_value, middle_value, right_value]
        """

        current_value = int(instance[feature_index])
        middle_value = math.ceil(current_value + noise)
        left_value_b = math.floor(float(left_value[::-1]))
        # print('left value is ' ,left_value)
        right_value_b = math.floor(float(right_value))
        # print('right value is ',right_value)
        coef = 2
        if int(right_value_b - left_value_b) > 2:
            while not (left_value_b < middle_value < right_value_b):
                if middle_value < left_value_b:  # for this to happen noise has to be a negative number
                    middle_value = current_value + (1 / coef * noise)
                    middle_value = math.ceil(middle_value)
                    coef += 1
                else:
                    middle_value = current_value - (1 / coef * noise)
                    middle_value = math.ceil(middle_value)
                    coef += 1
        else:
            middle_value = 'impossible'
        if left_value_b > min_value:
            left_value = int(left_value_b - 1)
        else:
            left_value = int(min_value)

        if right_value_b < max_value:
            right_value = int(right_value_b + 1)
        else:
            right_value = int(max_value)

        return [left_value, middle_value, right_value]

    def calc_single_feature_continuous_values(self, instance, feature_index, left_value, right_value, noise, min_value,
                                              max_value):
        """calc_single_feature_continuous_values finds and returns, for a single *continuous* feature, the following values:
            a) One value just below the lower bound of that feature in the LF rule.
            b) One value just above the higher bound of that feature in the LF rule.
            c) One value (for now, probably gonna be 2 for some features) which belongs in the middle of the previous mentioned bounds.

            This function ignores features that do not have ranges in the rule (transformed categorical).

        Args:
            instance: the instance for which the lf rule was made.
            feature_index: the index of the feature we need to calculate the values for in the feature names list.
            left_value: the lower bound of the feature in the lf rule.
            right_value: the higher bound of the feature in the lf rule.
            noise: the gaussian noise, calculated using the mean and std of the feature.
            min_value: the minimum accepted value for the feature.
            max_value: the maximum accepted value for the feature.
        Return:
            a list of values as [left_value, middle_value, right_value]
        """
        current_value = float(instance[feature_index])
        left_value_b = float(left_value[::-1])
        # print('left value is ' ,left_value_b)
        right_value_b = float(right_value)
        noise = noise_middle = abs(noise)

        while (right_value_b - left_value_b) <= (noise_middle / 2):
            if right_value_b < 1:
                noise_middle *= 0.5
            elif right_value_b >= 1:
                noise_middle /= 10

        if (current_value - left_value_b) > (right_value_b - current_value):
            middle_value = current_value - noise_middle / 2
        else:
            middle_value = current_value + noise_middle / 2

        coef = 2
        if left_value_b != min_value:
            left_value = float(left_value_b - abs(noise))
            while not (left_value > min_value):
                left_value = round(left_value_b - (1 / coef) * abs(noise), 4)
                coef += 1
        else:
            left_value = float(min_value)

        coef = 2
        if right_value_b != max_value:
            right_value = float(right_value_b + abs(noise))
            while not (right_value <= max_value):
                right_value = round(right_value_b + (1 / coef) * abs(noise), 4)
                coef += 1
        else:
            right_value = float(max_value)

        # keeping four digits for now.
        right_value = "{:.4f}".format(right_value)
        left_value = "{:.4f}".format(left_value)
        middle_value = "{:.4f}".format(middle_value)
        # print(right_value,middle_value,right_value)
        return [left_value, middle_value, right_value]

    def calculate_multiple_feature_values(self, instance, lf_rule, discrete_features):
        """calculate_multiple_feature_values composes and returns, for all the features contained in the lf rule,
            a dictionary in the form of feature(key)->values_list(value), with the values_list containing the following values:
            a) One value just below the lower bound of that feature in the LF rule.
            b) One value just above the higher bound of that feature in the LF rule.
            c) One value (for now, probably gonna be 2 for some features) which belongs in the middle of the previous mentioned bounds.

            This function ignores features that do not have ranges in the rule (transformed categorical).

        Args:
            instance: the instance for which the lf rule was made.
            lf_rule: the lf rule.
            discrete_features: a list containing all the discrete features.
        Return:
            a dictionary of features and the corresponding potentially new values.
        """
        dim = len(instance)
        rule = ','.join(str(v) for v in lf_rule)
        new_feature_values = {}
        for feat in range(dim):
            new_feature_values[feat] = []
        for feat in range(dim):
            feat_name = self.feature_names[feat]
            if rule.find(feat_name) != -1:
                '''
                if self.categorical_features is not None:
                    # temporary
                    if feat_name in self.categorical_features:
                        new_feature_values[feat] = []
                        continue
                '''
                new_feature_values[feat] = self.calculate_single_feature_values(instance, feat_name, rule,
                                                                                discrete_features)
        return new_feature_values

    def check_changes_in_prediction(self, instance, lf_rule, discrete_features):
        """check_changes_in_prediction composes and returns, for all the features contained in the lf rule, a dictionary in the
            form of feature(key) -> values_list(value), with the values list containing the values that change
            the model's prediction (change meaning getting a different prediction from the lf rule one) for a specific instance.
        Args:
            instance: the instance for which the lf rule was made.
            lf_rule: the lf rule.
            discrete_features: a list containing all the discrete features.
        Return:
            a dictionary of features and the corresponding values that change the model's prediction.
        """
        new_feature_values = self.calculate_multiple_feature_values(instance, lf_rule, discrete_features)
        not_included = []
        print('---------FEATURES THAT GOT REDUCED FROM LF BELOW---------')
        for f, v in new_feature_values.items():
            if not v:
                not_included.append(f)
                print(self.feature_names[f])

        for f in not_included:
            new_feature_values.pop(f)

        print('---------NEW FEATURE VALUES BELOW [left,middle,right]---------')
        for f, v in new_feature_values.items():
            print(self.feature_names[f], v)

        print('---------FEATURE VALUES THAT MAY CHANGE THE CLASSIFICATION BELOW---------')
        original_prediction = self.get_class_from_rule(lf_rule)
        probabilities_before = self.model.predict_proba([self.utilizer.transform([instance])[0]])[0]
        probabilities_before_calibrated = self.calibrator.predict_proba([self.utilizer.transform([instance])[0]])[
            0]  # Line 167.
        # print('original',probabilities_before,int(self.model.predict([self.utilizer.transform([instance])[0]])[0]))
        # new_prediction = -1
        new_instance = instance.copy()
        # changes_in_prediction = {}
        changes_in_probabilities = {}

        for f, v in new_feature_values.items():
            # changes_in_prediction[f] = []
            changes_in_probabilities[f] = []
            new_instance = instance.copy()
            for value in v:
                if value != 'impossible':
                    new_instance[self.feature_names.index(self.feature_names[f])] = value
                else:
                    new_instance[self.feature_names.index(self.feature_names[f])] = 0
                if self.utilizer is not None:
                    new_instance = self.utilizer.transform([new_instance])[0]

                # new_prediction = self.feature_names.index(int(self.model.predict([new_instance])[0]))
                probabilities_after = self.model.predict_proba([new_instance])[0]
                probabilities_after_calibrated = self.calibrator.predict_proba([new_instance])[0]
                # print(probabilities_after,new_prediction) #Just for Debugging

                if original_prediction == 0:
                    difference_in_proba = abs(probabilities_after[1] - probabilities_before[1])
                    difference_in_proba_calibrated = probabilities_after_calibrated[1] - \
                                                     probabilities_before_calibrated[1]

                    if difference_in_proba_calibrated > 0.05:  # keep this at 5% for now. Maybe adjust, or even automatically optimize it somehow in the future.
                        if (value != 'impossible'):
                            changes_in_probabilities[f].append(value)
                            changes_in_probabilities[f].append(original_prediction)
                            # changes_in_probabilities[f].append(new_prediction)
                            changes_in_probabilities[f].append(probabilities_before_calibrated[1])
                            changes_in_probabilities[f].append(probabilities_after_calibrated[1])
                            changes_in_probabilities[f].append(difference_in_proba)
                            changes_in_probabilities[f].append(difference_in_proba_calibrated)

                else:
                    difference_in_proba = abs(probabilities_after[0] - probabilities_before[0])
                    difference_in_proba_calibrated = probabilities_after_calibrated[0] - \
                                                     probabilities_before_calibrated[0]

                    if difference_in_proba_calibrated > 0.05:
                        if (value != 'impossible'):
                            changes_in_probabilities[f].append(value)
                            changes_in_probabilities[f].append(original_prediction)
                            # changes_in_probabilities[f].append(new_prediction)
                            changes_in_probabilities[f].append(probabilities_before_calibrated[0])
                            changes_in_probabilities[f].append(probabilities_after_calibrated[0])
                            changes_in_probabilities[f].append(difference_in_proba)
                            changes_in_probabilities[f].append(difference_in_proba_calibrated)
                new_instance = instance.copy()

        return changes_in_probabilities

    def extract_feature_statistics(self, train_data):
        """extract_feature_statistics composes and returns, working on the training data, a dictionary in the
            form of feature(key) -> values_list(value),
            with the values_list containing the minimum and maximum values of the feature, as well as its mean and std.
        Args:
            train_data: the training data.
        Return:
            a dictionary of features and the corresponding statistics.
        """
        dim = len(train_data[0])
        features_statistics = {}
        for feature in range(dim):
            features_statistics[feature] = []
        for feature in range(dim):
            features_statistics[feature].append(
                train_data[:, feature:feature + 1].min())
            features_statistics[feature].append(
                train_data[:, feature:feature + 1].max())
            features_statistics[feature].append(
                train_data[:, feature:feature + 1].mean())
            features_statistics[feature].append(
                train_data[:, feature:feature + 1].std())
        return features_statistics

    def get_class_from_rule(self, lf_rule):
        """get_class_from_rule finds and returns the classification result (class name) of a lf rule.
        Args:
            train_data: the training data.
        Return:
            the class name (int form).
        """

        rule = ','.join(str(v) for v in lf_rule)
        index = rule.find('then') + len('then') + 1
        class_name = ""
        while rule[index] != ",":
            class_name += rule[index]
            index += 1
        return self.class_names.index(class_name)

    def find_first_digit_index(self, feature, rule, direction):
        """find_first_digit_index finds and returns the index of the first digit to the left or to the right
            of a feature in a lf rule.
        Args:
            feature: the feature.
            rule: the lf rule.
            direction: the direction we need to move to and find the first digit.
        Return:
            the index of the first digit or -1 if there's no digit, meaning there are no ranges in the rule for that feature.
        """

        index = rule.find(feature)

        # no digits -> no range
        if rule[index - 1] == " ":
            return -1

        if direction == 'right':
            index += len(feature)
            while not rule[index].isdigit():
                index += 1
        else:  # direction is left
            while not rule[index].isdigit():
                index -= 1
        return index
