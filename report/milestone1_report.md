# Milestone 1: Formula 1 podium prediction

## What this report answers

The prediction is whether a driver finishes on the podium. The cutoff is after qualifying and before the race starts. Current-race results, points, laps, status, pit stops, and the standings published for that race are not features. Primary metric: PR-AUC, because a podium is the minority outcome. Secondary metric: F1 at a threshold chosen on the validation years.

## Question 1. Front-row conversion by circuit

Using the starting grid, the highest front-row conversion before 2014 is Circuit de Nevers Magny-Cours at 80.6% (29 podiums from 36 front-row starts). From 2014 onward it is Yas Marina Circuit at 86.4% (19 from 22). Circuits need at least 20 front-row starts in an era, so a handful of races cannot top the list. Among circuits that clear that bar in both eras, Circuit de Monaco gains the most ranks (23 to 2), and Hungaroring loses the most (4 to 10). Qualifying position is not the same list. In 2003-2024, 38 drivers qualified on the front row and started further back. They podium 28.9% of the time, against 72.6% for the 818 who qualified on the front row and started there. A penalty is why those two columns come apart, so the ranking above uses the grid.

## Question 2. Home races, controlling for the grid

The comparison holds the starting position roughly fixed, so a home driver who also qualified at the front is not the whole result. grid 1: home 57.6% (92 starts), away 64.7% (1032 starts), gap -7.1 points | grid 2-3: home 53.5% (172 starts), away 51.0% (2074 starts), gap +2.5 points | grid 4-10: home 15.1% (649 starts), away 16.6% (7231 starts), gap -1.5 points | grid 11+: home 1.6% (1411 starts), away 1.7% (12372 starts), gap -0.0 points. Blank grids and grid 0 are left out, because those rows have no starting slot to control for. East German nationality is matched to German circuits, and a dual national is at home in either country. This is a difference inside a grid band, not a claim that the crowd causes the result.

## Question 3. Mechanical retirements

A mechanical retirement is a start whose status names a failed component. Crashes, driver causes, regulation decisions, and unclassified 'Retired' rows are not counted. Cars that did not start are out of the denominator. Constructors with fewer than 30 starts in an era are left out of that era, and the comparison below keeps constructors that clear the bar in both. Highest rate in 2014-2021: McLaren at 14.5%. Highest rate in 2022-2024: Alfa Romeo at 12.5%. Largest rise: Alfa Romeo (+8.3 points). Largest fall: McLaren (-10.8 points).

## Features that are legal at the cutoff

Grid and front row are known once penalties are applied. Qualifying position and reaching Q3 are known when qualifying ends. Form uses the previous five starts only. Entering standings are the championship table after the previous round, so round 1 is blank on purpose. Home race compares the driver's nationality with the circuit country. Same-weekend sprint points are left out: they are known before the race in the current format, but they sit in a results table from the same event, and the safer reading of the brief is to exclude them. The model years are 2003-2024. Before 2003 the qualifying table does not cover every completed race, so the cutoff is not in the file.

## Split

Train 2003-2018, validation 2019-2021, test 2022-2024. The split is by season so a later race cannot be used to fit a model that is then scored on an earlier race. The 2022 regulation change sits inside the test years. Thresholds and hyperparameters are chosen on validation only.

## Model comparison

The selected model is random_forest, with parameters {'max_depth': 12}. It has the highest validation PR-AUC, 0.738. Its F1 threshold, 0.68, was chosen on the validation years and then frozen. On the test years its PR-AUC is 0.692 and its F1 is 0.654. Logistic regression is the baseline. On the test years its PR-AUC is 0.703 and its F1 is 0.653. The neural network is a shallow feed-forward network with one or two hidden layers. On the test years, neural_network has a higher PR-AUC (0.712) than the selected model (0.692). That fact is reported and is not used to change the choice.

              model      split precision recall    f1 roc_auc pr_auc threshold
logistic_regression      train     0.605  0.714 0.655   0.929  0.679      0.75
logistic_regression validation     0.662  0.783 0.718   0.929  0.721      0.75
logistic_regression       test     0.615  0.696 0.653   0.935  0.703      0.75
      random_forest      train     0.715  0.816 0.762   0.972  0.844      0.68
      random_forest validation     0.695  0.761 0.727   0.933  0.738      0.68
      random_forest       test     0.633  0.676 0.654   0.932  0.692      0.68
  gradient_boosting      train     0.614  0.802 0.696   0.947  0.736      0.72
  gradient_boosting validation     0.670  0.800 0.729   0.933  0.725      0.72
  gradient_boosting       test     0.602  0.681 0.639   0.918  0.675      0.72
     neural_network      train     0.566  0.800 0.663   0.936  0.703      0.72
     neural_network validation     0.635  0.822 0.717   0.931  0.734      0.72
     neural_network       test     0.548  0.789 0.647   0.932  0.712      0.72

## Preprocessing order

Both experiments use logistic regression with C=1. Only the order of imputation and scaling changes. The distribution table is the driver's entering championship position on the training years.

                            order      split   pr_auc  roc_auc
                impute then scale validation 0.720984 0.928715
                impute then scale       test 0.702749 0.935354
scale observed values then impute validation 0.720867 0.928671
scale observed values then impute       test 0.702624 0.935341

                          version  missing    median       min       max
           raw, blanks left empty      474 11.000000  1.000000 27.000000
                impute then scale        0 -0.049078 -1.648677  2.510279
scale observed values then impute        0 -0.051005 -1.590482  2.412159

Higher validation PR-AUC: impute then scale at 0.721.

## Ablation

Each row removes one feature group and retrains random_forest with the same hyperparameters. The largest validation drop is grid (0.035).  Removing form raises validation PR-AUC, so that group is not adding information once the other groups are present. The test column is a check. It is not used to pick the group.

        model removed_group  validation_pr_auc      drop  test_pr_auc
random_forest        (none)           0.737713  0.000000     0.692448
random_forest          grid           0.702580  0.035133     0.687959
random_forest    qualifying           0.713402  0.024310     0.702126
random_forest          form           0.745964 -0.008251     0.680020
random_forest     standings           0.727358  0.010355     0.677914
random_forest          home           0.731982  0.005731     0.700186

## Explanations are not causes

Permutation importance shuffles one feature of the selected model and measures the drop in validation PR-AUC. SHAP is computed for random_forest, because that model is a tree model and the values stay in the original feature units. The local LIME explanation is one test row: race 1114, driver 830, predicted probability 0.980, actual podium 1. A large weight means the model used that feature for this row. It does not mean the feature caused the podium.

## Inference on a complete row and a partial row

                                        record  driverId  raceId  actual_podium  podium_probability  predicted_podium  threshold
                             complete test row       830    1114           True            0.980220              True      0.675
same row with qualifying and standings removed       830    1114           True            0.644894             False      0.675

The second row is the same driver and race with qualifying position and entering standings removed. The pipeline fills those blanks from the training median. The probability changes, and the function still returns a score. On this run the yes-or-no decision also changes, because the probability crosses the frozen threshold.

## Limitations

Qualifying is missing or partial before 2003, so those seasons are not in the model. Q3 did not exist in the early seasons that are in the training window, and reached_q3 is left blank for races that had no Q3. Entering standings ignore sprint points earned earlier in the same weekend. The mechanical-retirement list follows the status text: a tyre or a wing counts only when the file names that part, and a generic Retired does not count. Home country is nationality, not birthplace. East German drivers are matched to German circuits, and a dual national matches either country. One 1983 row disagrees between positionOrder and the official position text; the label follows positionOrder, as the brief defines it. The test seasons are a different regulation era from most of the training seasons, so a drop from validation to test is part of the result, not a broken split.
