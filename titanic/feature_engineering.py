import numpy as np
import pandas as pd
from sklearn import model_selection
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor


# 特征分析
def feature_engineering(train_data, test_data):
    # 添加Family_size特征
    train_data["Family_size"] = train_data["SibSp"] + train_data["Parch"] + 1
    test_data["Family_size"] = test_data["SibSp"] + test_data["Parch"] + 1


    # 添加Fname特征，即姓氏
    train_data["Fname"] = train_data.Name.apply(lambda x: x.split(",")[0])
    test_data["Fname"] = test_data.Name.apply(lambda x: x.split(",")[0])



    # train_data[train_data["Family_size"]>0].sort_values(by=["Ticket"])
    # 结论：
    # 1、Fname相同，Ticket相同，Family_size>0可以确定一个家庭，只有一个家庭例外，这个家庭的票号连号，但影响不大。
    # 2、有女性死亡的家庭，除了1岁以下的婴儿外，家庭成员全部死亡。（分数第一次有大的提升，也是添加了该特征之后）
    # 3、家庭中若有大于18岁男性存活，或年龄为nan的男性存活，则该家庭全部存活。（加入该特征后，公分达到了0.8）
    # 4、其余家庭，除极个别外，符合男死女活小孩活
    # 5、Ticket相同的乘客，也许是朋友，大部分也符合以上3个家庭特征，但训练集中除去家庭成员后，同一Ticket的乘客较少，大部分符合男死女活小孩活。


    # 添加女性死亡家庭特征
    dead_train = train_data[train_data["Survived"] == 0]
    fname_ticket = dead_train[(dead_train["Sex"] == "female") & (dead_train["Family_size"] >= 1)][["Fname", "Ticket"]]
    train_data["dead_family"] = np.where(
        train_data["Fname"].isin(fname_ticket["Fname"]) & train_data["Ticket"].isin(fname_ticket["Ticket"]) & (
                    (train_data["Age"] >= 1) | train_data.Age.isnull()), 1, 0)
    test_data["dead_family"] = np.where(
        test_data["Fname"].isin(fname_ticket["Fname"]) & test_data["Ticket"].isin(fname_ticket["Ticket"]) & (
                    (test_data["Age"] >= 1) | test_data.Age.isnull()), 1, 0)


    # 添加男性存活家庭特征
    live_train = train_data[train_data["Survived"] == 1]
    live_fname_ticket = live_train[(live_train["Sex"] == "male") & (live_train["Family_size"] >= 1) & (
                (live_train["Age"] >= 18) | (live_train["Age"].isnull()))][["Fname", "Ticket"]]
    train_data["live_family"] = np.where(
        train_data["Fname"].isin(live_fname_ticket["Fname"]) & train_data["Ticket"].isin(live_fname_ticket["Ticket"]),
        1, 0)
    test_data["live_family"] = np.where(
        test_data["Fname"].isin(live_fname_ticket["Fname"]) & test_data["Ticket"].isin(live_fname_ticket["Ticket"]), 1,
        0)


    # 添加男死女活小孩活家庭特征
    dead_man_fname_ticket = train_data[
        (train_data["Family_size"] >= 1) & (train_data["Sex"] == "male") & (train_data["Survived"] == 0) & (
                    train_data["dead_family"] == 0)][["Fname", "Ticket"]]
    train_data["deadfamily_man"] = np.where(
        train_data["Fname"].isin(dead_man_fname_ticket["Fname"]) & train_data["Ticket"].isin(
            dead_man_fname_ticket["Ticket"]) & (train_data.Sex == "male"), 1, 0)
    train_data["deadfamily_woman"] = np.where(
        train_data["Fname"].isin(dead_man_fname_ticket["Fname"]) & train_data["Ticket"].isin(
            dead_man_fname_ticket["Ticket"]) & (train_data.Sex == "female"), 1, 0)
    test_data["deadfamily_man"] = np.where(
        test_data["Fname"].isin(dead_man_fname_ticket["Fname"]) & test_data["Ticket"].isin(
            dead_man_fname_ticket["Ticket"]) & (test_data.Sex == "male"), 1, 0)
    test_data["deadfamily_woman"] = np.where(
        test_data["Fname"].isin(dead_man_fname_ticket["Fname"]) & test_data["Ticket"].isin(
            dead_man_fname_ticket["Ticket"]) & (test_data.Sex == "female"), 1, 0)
    train_data.loc[
        (train_data["dead_family"] == 0) & (train_data["live_family"] == 0) & (train_data["deadfamily_man"] == 0) & (
                    train_data["deadfamily_woman"] == 0) & (train_data["Family_size"] >= 1) & (
                    train_data["Sex"] == "male"), "deadfamily_man"] = 1
    train_data.loc[
        (train_data["dead_family"] == 0) & (train_data["live_family"] == 0) & (train_data["deadfamily_man"] == 0) & (
                    train_data["deadfamily_woman"] == 0) & (train_data["Family_size"] >= 1) & (
                    train_data["Sex"] == "female"), "deadfamily_woman"] = 1
    test_data.loc[
        (test_data["dead_family"] == 0) & (test_data["live_family"] == 0) & (test_data["deadfamily_man"] == 0) & (
                    test_data["deadfamily_woman"] == 0) & (test_data["Family_size"] >= 1) & (
                    test_data["Sex"] == "male"), "deadfamily_man"] = 1
    test_data.loc[
        (test_data["dead_family"] == 0) & (test_data["live_family"] == 0) & (test_data["deadfamily_man"] == 0) & (
                    test_data["deadfamily_woman"] == 0) & (test_data["Family_size"] >= 1) & (
                    test_data["Sex"] == "female"), "deadfamily_woman"] = 1


    # 添加同票号男死女活小孩活特征
    grp_tk = train_data.drop(["Survived"], axis=1).append(test_data).groupby(["Ticket"])
    tickets = []
    for grp, grp_train in grp_tk:
        ticket_flag = True
        if len(grp_train) != 1:
            for i in range(len(grp_train) - 1):
                if grp_train.iloc[i]["Fname"] != grp_train.iloc[i + 1]["Fname"]:
                    ticket_flag = False
        if ticket_flag == False:
            tickets.append(grp)
    train_data.loc[(train_data.Ticket.isin(tickets)) & (train_data.Family_size == 0) & (
                train_data.Sex == "male"), "deadfamily_man"] = 1
    train_data.loc[(train_data.Ticket.isin(tickets)) & (train_data.Family_size == 0) & (
                train_data.Sex == "female"), "deadfamily_woman"] = 1
    test_data.loc[(test_data.Ticket.isin(tickets)) & (test_data.Family_size == 0) & (
                test_data.Sex == "male"), "deadfamily_man"] = 1
    test_data.loc[(test_data.Ticket.isin(tickets)) & (test_data.Family_size == 0) & (
                test_data.Sex == "female"), "deadfamily_woman"] = 1

    # 补充缺失的票号
    test_data = test_data.fillna({"Fare": test_data[test_data["Pclass"] == 3]["Fare"].mean()})


    # Sex特征处理
    train_data["Embarked"] = pd.factorize(train_data["Embarked"])[0]
    test_data["Embarked"] = pd.factorize(test_data["Embarked"])[0]
    train_data["Sex"] = pd.factorize(train_data["Sex"])[0]
    test_data["Sex"] = pd.factorize(test_data["Sex"])[0]
    # train_dummies_sex = pd.get_dummies(train_data["Sex"])
    # test_dummies_sex = pd.get_dummies(test_data["Sex"])
    # train_data = pd.concat([train_data, train_dummies_sex], axis=1)
    # test_data = pd.concat([test_data, test_dummies_sex], axis=1)

    # Name处理
    # 提取出Name中的称呼，对预测年龄有一定的帮助，对生存预测的帮助好像不大。
    train_name = train_data.Name.str.extract("([a-zA-Z]+)\.")
    test_name = test_data.Name.str.extract("([a-zA-Z]+)\.")
    train_name["Title"] = train_data.Name.str.extract("([a-zA-Z]+)\.")
    test_name["Title"] = test_data.Name.str.extract("([a-zA-Z]+)\.")

    train_name = train_name.drop([0], axis=1)
    test_name = test_name.drop([0], axis=1)

    train_name["Title"] = train_name["Title"].replace(["Mlle", "Ms"], "Miss")
    train_name["Title"] = train_name["Title"].replace(["Mme"], "Mrs")
    train_name["Title"] = train_name["Title"].replace(["Countess", "Sir", "Lady", "Don"], "Royal")
    train_name["Title"] = train_name["Title"].replace(["Dr", "Rev", "Col", "Major", "Jonkheer", "Capt"], "Rare")

    test_name["Title"] = test_name["Title"].replace(["Ms"], "Miss")
    test_name["Title"] = test_name["Title"].replace(["Dona"], "Mrs")
    test_name["Title"] = test_name["Title"].replace(["Dr", "Rev", "Col"], "Rare")

    train_name["Title"] = train_name["Title"].replace(["Mr"], 1)
    train_name["Title"] = train_name["Title"].replace(["Miss"], 2)
    train_name["Title"] = train_name["Title"].replace(["Mrs"], 3)
    train_name["Title"] = train_name["Title"].replace(["Master"], 4)
    train_name["Title"] = train_name["Title"].replace(["Royal"], 5)
    train_name["Title"] = train_name["Title"].replace(["Rare"], 6)

    test_name["Title"] = test_name["Title"].replace(["Mr"], 1)
    test_name["Title"] = test_name["Title"].replace(["Miss"], 2)
    test_name["Title"] = test_name["Title"].replace(["Mrs"], 3)
    test_name["Title"] = test_name["Title"].replace(["Master"], 4)
    test_name["Title"] = test_name["Title"].replace(["Rare"], 6)

    train_data["Title"] = train_name["Title"]
    test_data["Title"] = test_name["Title"]

    train_data = train_data.drop(["Name"], axis=1)
    test_data = test_data.drop(["Name"], axis=1)


    # 票价分段
    fare_bins = [-0.1, 7.854, 10.5, 21.558, 41.579, 512.5]
    train_data["FareBin"] = pd.cut(train_data["Fare"], fare_bins, labels=range(5))
    test_data["FareBin"] = pd.cut(test_data["Fare"], fare_bins, labels=range(5))


    train_data = train_data.drop(["SibSp", "Parch"], axis=1)
    test_data = test_data.drop(["SibSp", "Parch"], axis=1)
    train_data = train_data.drop(["PassengerId", "Ticket", "Cabin", "Embarked", "Fname"], axis=1)
    test_data = test_data.drop(["PassengerId", "Ticket", "Cabin", "Embarked", "Fname"], axis=1)
    # 年龄缺失值预测
    # 由于使用算法预测，每次预测结果有偏差，也会导致最终生存预测结果有偏差。
    age_train = pd.concat([train_data.drop(["Survived"], axis=1), test_data], axis=0)
    age_train = age_train[age_train["Age"].notnull()]
    
    age_label = age_train["Age"]
    age_train = age_train.drop(["Age"], axis=1)
    
    RFR = RandomForestRegressor(max_depth=16, n_estimators=16)
    RFR.fit(age_train, age_label)
    
    train_data.loc[train_data.Age.isnull(), ["Age"]] = RFR.predict(
        train_data[train_data.Age.isnull()].drop(["Age", "Survived"], axis=1))
    test_data.loc[test_data.Age.isnull(), ["Age"]] = RFR.predict(
        test_data[test_data.Age.isnull()].drop(["Age"], axis=1))
    # fill_missing_age(train_data)
    # fill_missing_age(test_data)


    # 年龄分段
    # 经肉眼观察数据，我发现没被我标记死亡家庭的15岁以下的小孩，基本存活，而年龄在50到80岁的男性死亡率超高。所以分段为：
    age_bins = [0, 15, 30, 49, 62, 80]
    train_data["AgeBin"] = pd.cut(train_data["Age"], age_bins, labels=range(5))
    test_data["AgeBin"] = pd.cut(test_data["Age"], age_bins, labels=range(5))

    # 删除多余特征
    train_data = train_data.drop(["Age", "Fare"], axis=1)
    test_data = test_data.drop(["Age", "Fare"], axis=1)

    return train_data, test_data


def fill_missing_age(df):
    missing_age_df = pd.DataFrame(df[["Age", "Embarked", "Sex", "Title", "Family_size", "Fare", "FareBin", "Pclass"]])
    # SettingWithCopyWarning: .copy()——追加
    missing_age_train = missing_age_df[missing_age_df["Age"].notnull()].copy()
    missing_age_test = missing_age_df[missing_age_df["Age"].isnull()].copy()
    df.loc[df["Age"].isnull(), "Age"] = get_missing_age(missing_age_train, missing_age_test)


# 建立Age的预测模型，我们可以多模型预测，然后再做模型的融合，提高预测的精度。
def get_missing_age(missing_age_train, missing_age_test):
    missing_age_X_train = missing_age_train.drop(["Age"], axis=1)
    missing_age_Y_train = missing_age_train["Age"]
    missing_age_X_test = missing_age_test.drop(["Age"], axis=1)

    gb_reg = GradientBoostingRegressor(random_state=42)
    gb_reg_param_grid = {"n_estimators": [2000], "max_depth": [4], "learning_rate": [0.01], "max_features": [3]}
    # iid=True——追加
    gb_reg_grid = model_selection.GridSearchCV(gb_reg, gb_reg_param_grid, iid=True, cv=10, n_jobs=25, verbose=1,
                                               scoring="neg_mean_squared_error")
    gb_reg_grid.fit(missing_age_X_train, missing_age_Y_train)
    print("Age Feature Best GB Params:" + str(gb_reg_grid.best_params_))
    print("Age Feature Best GB Score:" + str(gb_reg_grid.best_score_))
    print(
        "GB Train Error For 'Age' Feature Regressor" + str(gb_reg_grid.score(missing_age_X_train, missing_age_Y_train)))
    missing_age_test.loc[:, "Age_GB"] = gb_reg_grid.predict(missing_age_X_test)
    print(missing_age_test["Age_GB"][:4])

    rf_reg = RandomForestRegressor()
    rf_reg_param_grid = {"n_estimators": [200], "max_depth": [5], "random_state": [0]}
    # iid=True——追加
    rf_reg_grid = model_selection.GridSearchCV(rf_reg, rf_reg_param_grid, iid=True, cv=10, n_jobs=25, verbose=1,
                                               scoring="neg_mean_squared_error")
    rf_reg_grid.fit(missing_age_X_train, missing_age_Y_train)
    print("Age Feature Best RF Params:" + str(rf_reg_grid.best_params_))
    print("Age Feature Best RF Score:" + str(rf_reg_grid.best_score_))
    print(
        "RF Train Error For 'Age' Feature Regressor" + str(rf_reg_grid.score(missing_age_X_train, missing_age_Y_train)))
    missing_age_test.loc[:, "Age_RF"] = rf_reg_grid.predict(missing_age_X_test)
    print(missing_age_test["Age_RF"][:4])

    print("shape1", missing_age_test["Age"].shape, missing_age_test[["Age_GB", "Age_RF"]].mode(axis=1).shape)
    missing_age_test.loc[:, "Age"] = np.mean([missing_age_test["Age_GB"], missing_age_test["Age_RF"]])
    print(missing_age_test["Age"][:4])

    missing_age_test.drop(["Age_GB", "Age_RF"], axis=1, inplace=True)
    return missing_age_test