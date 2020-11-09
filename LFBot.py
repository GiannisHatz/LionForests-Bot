from flask import Flask, render_template, request, jsonify
import pandas as pd


class LFBot:
    def __init__(self, X, y, feature_names, categorical_features, class_names, parameters, description, lf,
                 discrete_features, categorical_map):
        self.categorical_map = categorical_map
        self.discrete_features = discrete_features
        self.timesCalled = 0
        self.lf_rule_original = []
        self.lf_rule = []
        self.instance_after_change = []
        self.original_instance = []
        self.feature_names = feature_names
        self.categorical_features = categorical_features
        self.class_names = class_names
        self.parameters = parameters
        self.description = description
        self.X = X
        self.y = y
        self.lf = lf
        self.feature_values = {}
        i = 0
        data_frame = pd.DataFrame(self.X, columns=self.feature_names)
        for feature in feature_names:
            self.feature_values[i] = []
            if feature in categorical_features:
                self.feature_values[i] = list(data_frame.apply(set)[feature_names.index(feature)])
                # for v in self.feature_values[i]:

                self.feature_values[i] = self.categorical_map[feature][1:]
            else:
                self.feature_values[i].append(lf.feature_statistics[feature_names.index(feature)][0])
                self.feature_values[i].append(lf.feature_statistics[feature_names.index(feature)][1])
            i += 1

        self.app = Flask(__name__)

        self.dialog_flow = {

            0: {
                'content': self.description
            },

            1: {
                'content': "Okay! Fill in the information please!",
            },

            2: {
                'content': "Here is the prediction and the explanation of LF: ",
            },
            3: {
                'content': "Would you like to inspect any feature from the rule?",
            },

            4: {
                'content': "Alright, which one?",
            },

            5: {
                'content': "Change one of the following values for ",
            },

            6: {
                'content': "Here is the prediction and the explanation of LF: ",
            },

            7: {
                'content': "Would you like to inspect any other feature before we continue?",
            },

            8: {
                'content': "Hmm, I see. Do you want to discard your previous changes to the instance?",
            },

            9: {
                'content': "Great! Which feature would you like to tweak this time?",
            },

            10: {
                'content': "Change one of the following values for ",
            },

            11: {
                'content': "Here is the prediction and the explanation of LF: ",
            },

            12: {
                'content': "I can see that there are some features you could tweak to maybe get a different prediction, "
                           "wanna check them?",
            },
            13: {
                'content': "Okay! Pick one!",
            },
            14: {
                'content': "Change one of the following values for",
            },
            15: {
                'content': "Here is the prediction and the explanation of LF: ",
            },
            16: {
                'content': "Thanks for your time, bye!",
            }
        }

        self.dialog_flow_index = 0

        self.mean_values = []
        for f in self.lf.feature_statistics:
            self.mean_values.append(self.lf.feature_statistics[f][2])

    def run(self):
        app = Flask(__name__)

        @app.route("/")
        def home():
            return render_template("index.html", features=self.feature_names, feature_values=self.feature_values,
                                   categorical_features=self.categorical_features, flow_index=self.dialog_flow_index,
                                   mean_values=self.mean_values, discrete_features=self.discrete_features,
                                   class_names=self.class_names, categorical_map=self.categorical_map)

        @app.route("/get")
        def get_bot_response():
            bot_text = self.dialog_flow.get(self.dialog_flow_index)['content']
            self.dialog_flow_index += 1
            return str(bot_text)

        @app.route("/getPrediction")
        def get_lf_prediction():
            self.instance_after_change = request.args.getlist('instance[]')
            if not self.instance_after_change:
                return 'failure'  # maybe render a failure.html or something
            else:
                self.accuracy = 0
                self.lf_rule = self.lf.following_breadcrumbs(self.instance_after_change, False, True, False,
                                                             complexity=4)
                if self.timesCalled == 0:
                    self.original_instance = self.instance_after_change
                    self.lf_rule_original = self.lf_rule
                    self.timesCalled += 1
                    if self.lf.utilizer is not None:
                        proba_calibrated = \
                            self.lf.calibrator.predict_proba([self.lf.utilizer.transform([self.original_instance])[0]])[
                                0]
                        if self.lf.get_class_from_rule(self.lf_rule_original) == 0:
                            self.accuracy = proba_calibrated[0]
                        else:
                            self.accuracy = proba_calibrated[1]
                return jsonify(rule=self.lf_rule[0], accuracy=self.accuracy)

        @app.route("/getNewFeatureValues")
        def get_new_feature_values():
            feature = request.args.get('feature')
            new_feature_values = self.lf.calculate_multiple_feature_values(self.original_instance, self.lf_rule,
                                                                           self.discrete_features)
            if not feature:
                return 'failure'  # maybe render a failure.html or something
            else:
                return jsonify(values=list(new_feature_values[self.feature_names.index(feature)]))

        @app.route("/getProbabilities")
        def get_probabilities():
            changes_in_probabilities = self.lf.check_changes_in_prediction(self.original_instance,
                                                                           self.lf_rule_original,
                                                                           self.discrete_features)
            return changes_in_probabilities

        @app.route("/restartSession")
        def restart_session():
            self.dialog_flow_index = 0
            self.timesCalled = 0
            return 'success'

        @app.route("/setDialogFlowIndex")
        def set_dialog_flow_index():
            index = request.args.get('flow_index')
            self.dialog_flow_index = int(index)
            return str(self.dialog_flow_index)

        app.run()
