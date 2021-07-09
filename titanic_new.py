#这一篇是优化泰坦尼克项目

#基本数据处理包引入
import pandas as pd
import numpy as np
#数据可视化包引入
import matplotlib.pyplot as plt
import seaborn as sns
#matplotlib默认设置显示不了符号，加上这一行
plt.rcParams['axes.unicode_minus']=False
#设置打印数据时可以最多显示1000列
pd.set_option('display.max_columns',1000)
#使用中文字体前，首先要设置，这里设置为楷体，下面字符串有中文的要在前面加个u
plt.rcParams['font.sans-serif'] = ['KaiTi']

#数据预处理包引入
import sklearn.preprocessing as preprocseeing

#机器学习模型引入
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor

#机器学习模型训练后进行效果评估的包（注意不是测试集测试）
from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

try:
    df_train= pd.read_csv('train.csv')
    df_test = pd.read_csv('test.csv')
    df_gender_submission = pd.read_csv('gender_submission.csv')
    print('Roading success!')
except:
    print('Roading fail!')


#数据分析有三个步骤
#counting correlation 和 integration
#首先是counting，探索性可视化，看数据
#简单条形图，密度图，对所有字段做单变量分析和与存活率关联的双变量分析
#看每个字段的缺失情况，不同取值是否和存活与否明显相关，相关就加入特征

#定义的这个函数做单变量分析图展示，有两个图很大非常耗内存
def counting_titanic_single():
    df_train.Pclass.value_counts().plot(kind = 'bar')
    plt.title(u'舱位等级分布图')
    plt.ylabel(u'人数')
    plt.xlabel(u'舱位等级，123分别由高到低')
    plt.show()
    #Pclass取值分析图，计数数据直接bar图

    df_train.Survived.value_counts().plot(kind = 'pie')
    plt.title(u'幸存者图')
    plt.ylabel(u'人数')
    plt.xlabel(u'是否获救,1是获救，0是未获救')
    plt.show()
    #幸存者分析图，技术数据，单这个主要看两边比例用饼图


    df_train.Age.plot(kind = 'density')
    plt.title(u'年龄分布图')
    plt.ylabel(u'人数')
    plt.xlabel(u'乘客年龄')
    plt.show()
    #年龄分布图，这是个连续变量，我们看连续变量的分部，用密度图或者柱状图
    #注意密度图X轴会有超出范围的刻度比如负数和更高的，不过这些不影响读图因为密度为0

    df_train.Embarked.value_counts().plot(kind='bar')
    plt.title(u'各登船口岸上船人数')
    plt.ylabel(u'人数')
    plt.xlabel(u'各个登船口岸')
    plt.show()
    #各口岸上船人数图，这个是记数数据，柱状图或者饼图都行

    #df_train.Fare.plot(kind='density')
    df_train.Fare.plot.density()
    plt.title(u'票价分布图')
    plt.ylabel(u'人数')
    plt.xlabel(u'票价区间')
    plt.xlim([-100,600])
    plt.xticks(range(-50,600,50))
    plt.show()
    #乘客票价分布图，连续变量，用密度图画
    #因为有较远的点，所以整个图需要设置一下，xticks读取x轴标签列表，这里设置50为一格
    #同样需要限制一下x轴范围，xlim上下限

    df_train.SibSp.value_counts().plot(kind='pie')
    plt.title(u'乘客的兄弟姐妹与配偶个数')
    plt.ylabel(u'兄弟姐妹与配偶个数')
    plt.xlabel(u'乘客')
    plt.show()
    #配偶个数看起来是连续，但实际上因为数据格式已经分类了，所以按照计数数据图分析，这里用饼图看比例

    df_train.Parch.value_counts().plot(kind='pie')
    plt.title(u'乘客的家长和孩子个数')
    plt.ylabel(u'家长和孩子个数')
    plt.xlabel(u'乘客')
    plt.show()
    #同上

    df_train.Embarked.value_counts().plot(kind='pie')
    plt.title(u'乘客从三个港口登船的分布')
    plt.ylabel(u'人数')
    plt.xlabel(u'港口名')
    plt.show()
    #同上

#定义的这个函数对几个变量分析与是否幸存的关系
def titanic_variable_correlation_Survived_analysis():
    #是否幸存是个分类问题，y轴是确定的1或者0

    #乘客舱等级和是否获救的关联性
    Pclass_Survived_1 = df_train.Pclass[df_train.Survived == 1].value_counts()
    Pclass_Survived_0 = df_train.Pclass[df_train.Survived == 0].value_counts()
    Pclass_correlation_Survived = pd.DataFrame({u'获救者':Pclass_Survived_1,u'遇难者':Pclass_Survived_0})
    Pclass_correlation_Survived.plot(kind = 'bar',stacked = True)  #Stack参数，堆积柱状图
    plt.title(u"各乘客等级的获救情况")
    plt.xlabel(u"乘客等级")
    plt.ylabel(u"人数")
    plt.show()

    #看看年龄与是否获救的关系，直接两张图看看获救的年龄分布和未获救的年龄分布
    #如果获救者与遇难者的分布几乎相同，那这个特征就用处不大了
    Age_Survived_1 = df_train.Age[df_train.Survived == 1]
    Age_Survived_0 = df_train.Age[df_train.Survived == 0]
    df_Age_Survived = pd.DataFrame({u'获救者年龄分布':Age_Survived_1,u'遇难者年龄分布':Age_Survived_0})
    df_Age_Survived.plot.kde()
    plt.title(u"各年龄的获救分布情况")
    plt.xlabel(u"乘客年龄")
    plt.ylabel(u"概率密度")
    plt.show()

    #同理我们看看性别与获救的关系
    Sex_Survived_1 = df_train.Sex[df_train.Survived == 1].value_counts()
    Sex_Survived_0 = df_train.Sex[df_train.Survived == 0].value_counts()
    df_Sex_Survived = pd.DataFrame({u'获救者性别比例': Sex_Survived_1, u'遇难者性别比例': Sex_Survived_0})
    df_Sex_Survived.plot(kind = 'bar',stacked = True)
    plt.title(u"各性别的获救情况")
    plt.xlabel(u"性别")
    plt.ylabel(u"人数")
    plt.show()

    #登船入口是否和获救有关，各口岸乘客存活情况
    df_Embarked_Survived_1 = df_train.Embarked[df_train.Survived == 1].value_counts()
    df_Embarked_Survived_0 = df_train.Embarked[df_train.Survived == 0].value_counts()
    df_Embarked_Survived = pd.DataFrame({u'获救者港口分布': df_Embarked_Survived_1, u'遇难者港口分布': df_Embarked_Survived_0})
    df_Embarked_Survived.plot(kind = 'bar',stacked = True)
    plt.title(u"各港口的获救情况")
    plt.xlabel(u"不同港口")
    plt.ylabel(u"人数")
    plt.show()

    #计算p相关性，绘制热力图
    df_train.iloc[:, [2, 4, 5, 9, 11, 1]].corr()
    sns.heatmap(df_train.iloc[:, [2, 4, 5, 9, 11, 1]].corr(), annot=True, fmt='.1f', cmap='rainbow')
    plt.show()
    #

def titanic_preparation_missing_value_completation():
    # 为了方便阅读所以弄得函数写法，但这里的函数只是为了给训练集用的，所以不传参数了直接修改全局变量
    global df_train
    #Age缺失177个，Cabin缺失687个，Embarked缺失两个

    #Age直接用均值填充（就是下面这一行）,不过大部分其他人处理方法是使用随机森林通过其他特征推断出来的
    #均值法df_train.Age.fillna(df_train.Age.mean(), inplace=True)
    #随机森林方法，完整信息有Fare，Parch，SibSp，Pclass，通过这几个预测Age
    df_train_age = df_train[['Age','Fare', 'Parch', 'SibSp', 'Pclass']]
    df_train_age_notnull = df_train_age.loc[(df_train['Age'].notnull())]
    df_train_age_isnull = df_train_age.loc[(df_train['Age'].isnull())]
    X = df_train_age_notnull.values[:,1:]
    Y = df_train_age_notnull.values[:,0]
    clf = RandomForestRegressor(n_estimators=1000,n_jobs=-1)
    clf.fit(X,Y)
    predict = clf.predict(df_train_age_isnull.values[:,1:])
    df_train.loc[df_train['Age'].isnull(),['Age']] = predict

    #Cabin缺失太多，简单方法就是下面的直接降维，改为有舱位号和无舱位号
    """
    df_train.loc[(df_train.Cabin.notnull()), 'Cabin'] = 'Yes'
    df_train.loc[(df_train.Cabin.isnull()), 'Cabin'] = 'No'
    print(df_train.Cabin)
    """
    #新方法是，有舱位号里面继续细分，保留更多信息，比如C123和C148，分为C组，没舱位号还是单独一组
    df_train.loc[(df_train.Cabin.isnull()), 'Cabin'] = 'Z'
    df_train_Cabin_notnull = df_train.loc[(df_train['Cabin'].notnull())]
    #转化为字符串类型并获取第一个字符串
    df_train_Cabin_notnull_str = df_train_Cabin_notnull.Cabin.astype(str).str[0]
    df_train.loc[(df_train.Cabin.notnull(),'Cabin')] = df_train_Cabin_notnull_str
    #转化后的Cabin就是ABCDEFGTZ单个字母分类

    #Embarked只缺少两个，那就随便填入两个
    df_train.loc[(df_train.Embarked.isnull(),'Embarked')] = 'S'
    print(df_train.isnull().sum())

def feature_engineering():
    global df_train
    #注意这个是新增的特征，所以应当在缺失数值填补后运行，且在数据规则化前运行
    # 年龄这个我们可以考虑再做个特征工程，离散化分为三类，按照我们常识孩子和老年人有限，我们分为孩子，中年和老年
    # 而成年人里面我们考虑到女士优先，分为男与女
    df_train['Age_sex_class'] = 'child'
    df_train['Age_sex_class'].loc[(df_train['Age'] > 18) & (df_train['Sex'] == 'male')] = 'man'
    df_train['Age_sex_class'].loc[(df_train['Age'] > 18) & (df_train['Sex'] == 'female')] = 'woman'
    df_train['Age_sex_class'].loc[(df_train['Age'] > 65)] = 'old'
    print(df_train.Age_sex_class)

    #画个图看一下
    Sex_Survived_1 = df_train.Age_sex_class[df_train.Survived == 1].value_counts()
    Sex_Survived_0 = df_train.Age_sex_class[df_train.Survived == 0].value_counts()
    df_Sex_Survived = pd.DataFrame({u'获救者': Sex_Survived_1, u'遇难者': Sex_Survived_0})
    df_Sex_Survived.plot(kind='bar', stacked=True)
    plt.title(u"各年龄性别的获救情况")
    plt.xlabel(u"年龄性别")
    plt.ylabel(u"人数")
    plt.show()

    #看论坛里面很多人提到过家庭规模，以及家庭里面是否包含某些成员关系很大，我们也把这些纳入特征
    #新增FamilySize特征
    df_train['FamilySize'] = df_train['SibSp'] + df_train['Parch'] + 1
    #新增是否有单独乘客特征，包括普通单独乘客，老人单独乘客（儿童单独乘客应该不存在如果有也应该是缺失了数据或者错误）
    df_train['IsAlone'] = 'NoAlone'
    df_train.loc[df_train['FamilySize'] == 1,'IsAlone']= 'IsAlone'
    df_train.loc[(df_train['FamilySize'] == 1) & (df_train['Age'] > 60), 'IsAlone'] = 'IsSeniorAlone'
    # 画个图看一下
    Sex_Survived_1 = df_train.IsAlone[df_train.Survived == 1].value_counts()
    Sex_Survived_0 = df_train.IsAlone[df_train.Survived == 0].value_counts()
    df_Sex_Survived = pd.DataFrame({u'获救者': Sex_Survived_1, u'遇难者': Sex_Survived_0})
    df_Sex_Survived.plot(kind='bar', stacked=True)
    plt.title(u"单独乘客获救情况")
    plt.xlabel(u"单独乘客")
    plt.ylabel(u"人数")
    plt.show()


    #新增家庭中成员是否有成年人，孩子老人
    df_train['Family_members'] = 'other'
    df_train.loc[(df_train['FamilySize'] > 1) & (df_train['Age'] < 18), 'Family_members'] = 'have_child'
    df_train.loc[(df_train['FamilySize'] > 1) & (df_train['Age'] > 18) & (df_train['Sex'] == 'male'), 'Family_members'] = 'have_male'
    df_train.loc[(df_train['FamilySize'] > 1) & (df_train['Age'] > 60), 'Family_members'] = 'have_senior'
    print(df_train.Family_members)
    # 画个图看一下
    Sex_Survived_1 = df_train.Family_members[df_train.Survived == 1].value_counts()
    Sex_Survived_0 = df_train.Family_members[df_train.Survived == 0].value_counts()
    df_Sex_Survived = pd.DataFrame({u'获救者': Sex_Survived_1, u'遇难者': Sex_Survived_0})
    df_Sex_Survived.plot(kind='bar', stacked=True)
    plt.title(u"不同家庭成员分布获救情况")
    plt.xlabel(u"家庭成员")
    plt.ylabel(u"人数")
    plt.show()

    #我认为名字很难有什么特征，但是如果是称呼有可能包含地位高低，
    #这里直接用了论坛里面一个比较方便的处理，先提取出来后，分为几大类，然后有一些可能是拼写错误的处理一下
    df_train['Title'] = df_train['Name'].str.extract('([A-Za-z]+)\.')
    df_train['Title'] = df_train['Title'].replace(['Lady', 'Countess', 'Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    df_train['Title'] = df_train['Title'].replace('Mlle', 'Miss')
    df_train['Title'] = df_train['Title'].replace('Ms', 'Miss')
    df_train['Title'] = df_train['Title'].replace('Mme', 'Mrs')


def titanic_preparation():
    #为了方便阅读所以弄得函数写法，但这里的函数只是为了给训练集用的，所以不传参数了直接修改全局变量
    global df_train
    #这里的预处理主要指的是数据规则化
    dummies_Embarked = pd.get_dummies(df_train['Embarked'], prefix='Embarked')
    dummies_Sex = pd.get_dummies(df_train['Sex'], prefix='Sex')
    dummies_Pclass = pd.get_dummies(df_train['Pclass'], prefix='Pclass')
    dummies_Age_sex_class = pd.get_dummies(df_train['Age_sex_class'],prefix='Age_sex_class')
    dummies_Cabin = pd.get_dummies(df_train['Cabin'], prefix='Cabin')
    dummies_FamilySize = pd.get_dummies(df_train['FamilySize'],prefix='FamilySize')
    dummies_IsAlone = pd.get_dummies(df_train['IsAlone'], prefix='IsAlone')
    dummies_Family_members = pd.get_dummies(df_train['Family_members'], prefix='Family_members')
    dummies_Title = pd.get_dummies(df_train['Title'], prefix='Title')
    #concat别忘了 axis=1，我们是增加的新列而不是增加的新行
    df_train = pd.concat([df_train,dummies_Embarked,dummies_Sex,dummies_Pclass,dummies_Cabin,dummies_Age_sex_class,dummies_FamilySize,dummies_IsAlone,dummies_Family_members,dummies_Title],axis = 1)
    df_train.drop(['Pclass','Name','Sex','Ticket','Cabin','Embarked','Age_sex_class','FamilySize','IsAlone','Family_members','Title'],axis = 1, inplace=True)
    #再执行数据归一化，因为age与fare数值比其他属性高一个数量级别
    #需要将其特征化到[-1,1]之间
    # 1.使用StandarScaler实例化，获得一个scaler
    scaler = preprocseeing.StandardScaler()
    # 进行归一化处理，但scaler只接受numpy参数，所以要转化为二维numpy(用reshape)
    Age_numpy = df_train['Age'].values
    Age_numpy_reshape = Age_numpy.reshape(-1, 1)
    # 将其传入一个dataframe里面作为新的一列
    df_train['Age_scaled'] = scaler.fit_transform(Age_numpy_reshape)

    Fare_numpy = df_train['Fare'].values
    Fare_numpy_reshape = Fare_numpy.reshape(-1, 1)
    df_train['Fare_scaled'] = scaler.fit_transform(Fare_numpy_reshape)

def titanic_decision_tree_training():
    global df_train
    # 用正则取出我们所需要的属性值
    train_df = df_train.filter(regex='Survived|Age_.*|Age_sex_class_.*|Cabin_.*|Title_.*|FamilySize_.*|IsAlone_.*|Family_members_.*|SibSp|Parch|Fare_.*|Embarked_.*|Sex_.*|Pclass_.*')
    train_np = train_df.values
    # 分割，把训练集分割为样本矩阵和分类矩阵，用矩阵存放
    Y_train_Survived = train_np[:, 0]
    X_train_feature = train_np[:, 1:]
    tree_clf = tree.DecisionTreeClassifier()
    clf = tree_clf.fit(X_train_feature,Y_train_Survived)

    return clf

def model_evaluating(clf):
    global df_train
    #这个函数是用来本地评估的，使用Cross Validation评估
    all_data = df_train.filter(regex='Survived|Age_.*|Age_sex_class_.*|Cabin_.*|Title_.*|FamilySize_.*|IsAlone_.*|Family_members_.*|SibSp|Parch|Fare_.*|Embarked_.*|Sex_.*|Pclass_.*')
    X = all_data.values[:, 1:]
    Y = all_data.values[:, 0]
    # K折交叉验证，这里需要的参数为，模型clf，X训练数据，Y标签数据，cv表示折叠10次
    evaluating_result_scores = cross_validate(clf, X, Y, cv=10,return_train_score = 1)
    print(evaluating_result_scores)
    print(np.mean(evaluating_result_scores['train_score']))
    print(np.mean(evaluating_result_scores['test_score']))

def Gaussian_Naive_Bayes_classification_Titanic():
    #高斯贝叶斯其实要求特征符合高斯分布，前面图片可以看出有部分符合高斯分布
    global df_train
    # 用正则取出我们所需要的属性值
    train_df = df_train.filter(regex='Survived|Age_.*|Age_sex_class_.*|Cabin_.*|Title_.*|FamilySize_.*|IsAlone_.*|Family_members_.*|SibSp|Parch|Fare_.*|Embarked_.*|Sex_.*|Pclass_.*')
    train_np = train_df.values
    # 分割，把训练集分割为样本矩阵和分类矩阵，用矩阵存放
    Y_train_Survived = train_np[:, 0]
    X_train_feature = train_np[:, 1:]
    gnb = GaussianNB()
    clf = gnb.fit(X_train_feature, Y_train_Survived)

    return clf

def Nearest_Neighbors_Classification_Titanic():
    #最近邻分类，对特征敏感
    global df_train
    # 用正则取出我们所需要的属性值
    train_df = df_train.filter(regex='Survived|Age_.*|Age_sex_class_.*|Cabin_.*|Title_.*|FamilySize_.*|IsAlone_.*|Family_members_.*|SibSp|Parch|Fare_.*|Embarked_.*|Sex_.*|Pclass_.*')
    train_np = train_df.values
    # 分割，把训练集分割为样本矩阵和分类矩阵，用矩阵存放
    Y_train_Survived = train_np[:, 0]
    X_train_feature = train_np[:, 1:]
    # n_neighbors设置很重要，网上说法很多，我就选默认的5吧
    neigh = KNeighborsClassifier(n_neighbors=5)
    clf = neigh.fit(X_train_feature, Y_train_Survived)

    return clf

def RandomForest_Titanic():
    #集成学习的随机森林分类器，这个适合多一些冗余属性
    global df_train
    # 用正则取出我们所需要的属性值
    train_df = df_train.filter(regex='Survived|Age_.*|Age_sex_class_.*|Cabin_.*|Title_.*|FamilySize_.*|IsAlone_.*|Family_members_.*|SibSp|Parch|Fare_.*|Embarked_.*|Sex_.*|Pclass_.*')
    print('检测一下数据')
    print(train_df)
    train_np = train_df.values
    # 分割，把训练集分割为样本矩阵和分类矩阵，用矩阵存放
    Y_train_Survived = train_np[:, 0]
    X_train_feature = train_np[:, 1:]
    clf = RandomForestClassifier(n_estimators = 55,max_depth = 11,random_state=90)
    clf = clf.fit(X_train_feature, Y_train_Survived)

    return clf
def RandomForest_parameter():
    # 调参，绘制学习曲线来调参n_estimators（对随机森林影响最大）
    score_lt = []
    # 每隔10步建立一个随机森林，获得不同n_estimators的得分
    # 这个函数是用来本地评估的，使用Cross Validation评估
    all_data = df_train.filter(
        regex='Survived|Age_.*|Age_sex_class_.*|Cabin_.*|Title_.*|FamilySize_.*|IsAlone_.*|Family_members_.*|SibSp|Parch|Fare_.*|Embarked_.*|Sex_.*|Pclass_.*')
    X = all_data.values[:, 1:]
    Y = all_data.values[:, 0]
    # K折交叉验证，这里需要的参数为，模型clf，X训练数据，Y标签数据，cv表示折叠10次
    # 下面第一次运行时，大概61颗子树最好，所以这里先注释掉，继续缩小参数范围
    """
    for i in range(0, 300, 10):
        rfc = RandomForestClassifier(n_estimators=i + 1
                                     , random_state=90)
        score = cross_val_score(rfc, X, Y, cv=10).mean()
        score_lt.append(score)
    score_max = max(score_lt)
    print('最大得分：{}'.format(score_max),
          '子树数量为：{}'.format(score_lt.index(score_max) * 10 + 1))
    # 绘制学习曲线
    x = np.arange(1, 300, 10)
    plt.subplot(111)
    plt.plot(x, score_lt, 'r-')
    plt.show()
    """
    # 在61附近缩小n_estimators的范围为51-71
    score_lt = []
    for i in range(51, 71):
        rfc = RandomForestClassifier(n_estimators=i
                                     , random_state=90)
        score = cross_val_score(rfc, X, Y, cv=10).mean()
        score_lt.append(score)
    score_max = max(score_lt)
    print('最大得分：{}'.format(score_max),
          '子树数量为：{}'.format(score_lt.index(score_max) + 30))

    # 绘制学习曲线
    x = np.arange(51, 71)
    plt.subplot(111)
    plt.plot(x, score_lt, 'o-')
    plt.show()
    # 子树55时最大

    #接下来减小复杂度调节深度
    # 建立n_estimators为45的随机森林
    rfc = RandomForestClassifier(n_estimators=55, random_state=90)

    # 用网格搜索调整max_depth
    param_grid = {'max_depth': np.arange(1, 20)}
    GS = GridSearchCV(rfc, param_grid, cv=10)
    GS.fit(X, Y)

    best_param = GS.best_params_
    best_score = GS.best_score_
    print(best_param, best_score)
    #结果为max_depth 11,提升了精确度
    #现在随机森林得最佳参数为 55,11

def test_preparation():
    #处理测试集数据，处理方法和训练集一样

    #1.缺失值填充
    #测试集缺失Age 86个，Fare 1个，Cabin 327个
    #Fare直接均值填充，Age还是用随机森林处理，Cabin处理方式相同
    global df_test
    print(df_test.isnull().sum())
    # Fare
    df_test.Fare.fillna(df_test.Fare.mean(), inplace=True)
    # Age
    df_test_age = df_test[['Age', 'Fare', 'Parch', 'SibSp', 'Pclass']]
    df_test_age_notnull = df_test_age.loc[(df_test['Age'].notnull())]
    df_test_age_isnull = df_test_age.loc[(df_test['Age'].isnull())]
    X = df_test_age_notnull.values[:, 1:]
    Y = df_test_age_notnull.values[:, 0]
    clf = RandomForestRegressor(n_estimators=1000, n_jobs=-1)
    clf.fit(X, Y)
    predict = clf.predict(df_test_age_isnull.values[:, 1:])
    df_test.loc[df_test['Age'].isnull(), ['Age']] = predict
    # Cabin
    df_test.loc[(df_test.Cabin.isnull()), 'Cabin'] = 'Z'
    df_test_Cabin_notnull = df_test.loc[(df_test['Cabin'].notnull())]
    df_test_Cabin_notnull_str = df_test_Cabin_notnull.Cabin.astype(str).str[0]
    df_test.loc[(df_test.Cabin.notnull(), 'Cabin')] = df_test_Cabin_notnull_str

    #2.和训练集相同，新的特征
    # Age_sex_class
    df_test['Age_sex_class'] = 'child'
    df_test['Age_sex_class'].loc[(df_test['Age'] > 18) & (df_test['Sex'] == 'male')] = 'man'
    df_test['Age_sex_class'].loc[(df_test['Age'] > 18) & (df_test['Sex'] == 'female')] = 'woman'
    df_test['Age_sex_class'].loc[(df_test['Age'] > 65)] = 'old'
    print(df_test.Age_sex_class)

    # FamilySize
    df_test['FamilySize'] = df_test['SibSp'] + df_test['Parch'] + 1

    # IsAlone
    df_test['IsAlone'] = 'NoAlone'
    df_test.loc[df_test['FamilySize'] == 1, 'IsAlone'] = 'IsAlone'
    df_test.loc[(df_test['FamilySize'] == 1) & (df_test['Age'] > 60), 'IsAlone'] = 'IsSeniorAlone'

    # Family_members
    df_test['Family_members'] = 'other'
    df_test.loc[(df_test['FamilySize'] > 1) & (df_test['Age'] < 18), 'Family_members'] = 'have_child'
    df_test.loc[(df_test['FamilySize'] > 1) & (df_test['Age'] > 18) & (
                df_test['Sex'] == 'male'), 'Family_members'] = 'have_male'
    df_test.loc[(df_test['FamilySize'] > 1) & (df_test['Age'] > 60), 'Family_members'] = 'have_senior'
    print(df_test.Family_members)

    # Title
    df_test['Title'] = df_test['Name'].str.extract('([A-Za-z]+)\.')
    df_test['Title'] = df_test['Title'].replace(
        ['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    df_test['Title'] = df_test['Title'].replace('Mlle', 'Miss')
    df_test['Title'] = df_test['Title'].replace('Ms', 'Miss')
    df_test['Title'] = df_test['Title'].replace('Mme', 'Mrs')

    #3.规则化
    dummies_Embarked = pd.get_dummies(df_test['Embarked'], prefix='Embarked')
    dummies_Sex = pd.get_dummies(df_test['Sex'], prefix='Sex')
    dummies_Pclass = pd.get_dummies(df_test['Pclass'], prefix='Pclass')
    dummies_Age_sex_class = pd.get_dummies(df_test['Age_sex_class'], prefix='Age_sex_class')
    dummies_Cabin = pd.get_dummies(df_test['Cabin'], prefix='Cabin')
    dummies_FamilySize = pd.get_dummies(df_test['FamilySize'], prefix='FamilySize')
    dummies_IsAlone = pd.get_dummies(df_test['IsAlone'], prefix='IsAlone')
    dummies_Family_members = pd.get_dummies(df_test['Family_members'], prefix='Family_members')
    dummies_Title = pd.get_dummies(df_test['Title'],prefix='Title')
    # concat别忘了 axis=1，我们是增加的新列而不是增加的新行
    df_test = pd.concat([df_test, dummies_Embarked, dummies_Title, dummies_Sex, dummies_Pclass, dummies_Cabin, dummies_Age_sex_class,
                          dummies_FamilySize, dummies_IsAlone, dummies_Family_members], axis=1)
    df_test.drop(['Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked', 'Age_sex_class', 'FamilySize', 'IsAlone',
                   'Family_members','Title'], axis=1, inplace=True)

    scaler = preprocseeing.StandardScaler()
    Age_numpy = df_test['Age'].values
    Age_numpy_reshape = Age_numpy.reshape(-1, 1)
    df_test['Age_scaled'] = scaler.fit_transform(Age_numpy_reshape)
    Fare_numpy = df_test['Fare'].values
    Fare_numpy_reshape = Fare_numpy.reshape(-1, 1)
    df_test['Fare_scaled'] = scaler.fit_transform(Fare_numpy_reshape)

    df_test = df_test.filter(
        regex='Age_.*|Age_sex_class_.*|Cabin_.*|Title_.*|FamilySize_.*|IsAlone_.*|Family_members_.*|SibSp|Parch|Fare_.*|Embarked_.*|Sex_.*|Pclass_.*')

    #因为训练集里面缺少了一种类型的Cabin，所以onehot会后缺少维度，这里补上一列
    df_test['Cabin_T'] = 0
    return df_test

def Model_evaluation(clf):
    global df_train
    #这个函数将模型中的特征重要性打印输出
    all_data = df_train.filter(
        regex='Survived|Age_.*|Age_sex_class_.*|Cabin_.*|Title_.*|FamilySize_.*|IsAlone_.*|Family_members_.*|SibSp|Parch|Fare_.*|Embarked_.*|Sex_.*|Pclass_.*')
    X = all_data.values[:, 1:]
    Y = all_data.values[:, 0]
    #获取模型输入特征的名字的矩阵
    X_name = all_data.columns[1:]
    #获取模型的特征重要性
    eva = clf.feature_importances_
    print('X_name')
    print(X_name)
    print('feature_importances')
    print(eva)
    #然后将前面两个矩阵映射为字典
    feature_map_dict = dict(zip(X_name,eva))
    #按照特征重要性大小排序输出
    print(sorted(feature_map_dict.items(), key = lambda kv:(kv[1], kv[0])))

#主程序全过程
#看看整体数据情况
#df_train.describe()
#变量分布做图，先看个各个变量大概
#counting_titanic_single()
#对变量分析一下和是否幸存的关系
#titanic_variable_correlation_Survived_analysis()
#查看数据缺失情况
#print(df_train.isnull().sum())
#补齐数据
titanic_preparation_missing_value_completation()
#添加新的特征
feature_engineering()
#数据预处理主要是标准化和one hot
titanic_preparation()
#看一下处理后的训练集情况
print(df_train)
#调用机器学习模型进行训练,因为直接用的全局变量，就没传参数
#调用函数后直接返回了

clf_decision_tree = titanic_decision_tree_training()
clf_Gaussion_Naive_Bayes = Gaussian_Naive_Bayes_classification_Titanic()
clf_Nearest_Neighbors = Nearest_Neighbors_Classification_Titanic()
clf_RandomForest = RandomForest_Titanic()

#交叉验证模型

model_evaluating(clf_Gaussion_Naive_Bayes)
print('以上是高斯朴素贝叶斯分类的交叉验证平均得分')
model_evaluating(clf_decision_tree)
print('以上是决策树分类的交叉验证平均得分')
model_evaluating(clf_Nearest_Neighbors)
print('以上是最近邻分类的交叉验证平均得分')
model_evaluating(clf_RandomForest)
print('以上是随机森林的交叉验证平均得分')

Model_evaluation(clf_RandomForest)
#随机森林调参：
#RandomForest_parameter()
#随机森林最佳参数n_estimators = 55,max_depth = 11

#测试集验证并生成预测结果
#修改df_test之前先保留其原始数据，因为有PassengerID
df_test_data = df_test
test = test_preparation()
print(test)
predictions = clf_RandomForest.predict(test)
result = pd.DataFrame({'PassengerId': df_test_data['PassengerId'].values, 'Survived': predictions.astype(np.int32)})
result.to_csv("RandomForest_predictions.csv", index=False)
"""
predictions = clf_Nearest_Neighbors.predict(test)
result = pd.DataFrame({'PassengerId': df_test_data['PassengerId'].values, 'Survived': predictions.astype(np.int32)})
result.to_csv("Nearest_Neighbors_predictions.csv", index=False)
"""