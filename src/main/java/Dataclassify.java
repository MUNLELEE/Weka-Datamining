import weka.attributeSelection.AttributeSelection;
import weka.attributeSelection.BestFirst;
import weka.attributeSelection.CfsSubsetEval;
import weka.attributeSelection.InfoGainAttributeEval;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.RandomForest;
import weka.core.Attribute;
import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.ConverterUtils;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Discretize;
import weka.filters.unsupervised.attribute.Normalize;
import weka.filters.unsupervised.attribute.NumericToNominal;
import weka.filters.unsupervised.attribute.Remove;

import java.io.File;
import java.io.FileReader;
import java.util.Random;

public class Dataclassify {
    private String filepath;
    private String newFilepath;
    private Instances data;
    private long startTime, endTime;  // 用来记录程序时间的变量
    private Instances[] store = new Instances[2];  // 用来保存训练集和测试集

    Dataclassify(String filepath) {
        this.filepath = filepath;
    }

    private Instances ChangeDataClass(Instances data) throws Exception {
        // 对数据进行类型转换的函数
        NumericToNominal ntn = new NumericToNominal();
        ntn.setInputFormat(data);
        data = Filter.useFilter(data, ntn);
        return data;
    }

    private Instances DiscretizeData(Instances data) throws Exception {
        Discretize dis = new Discretize();
        dis.setInputFormat(data);
        data = Filter.useFilter(data, dis);
        return data;
    }

    // 将类中的Instances进行实例化
    private void ConfirmData(String filepath) throws Exception {
        data = new Instances(new FileReader(filepath));
    }

    // 转换文件类型函数，将csv文件转换为arff文件可以方便使用
    public void Csv2Arff() throws Exception {
        Instances data = ConverterUtils.DataSource.read(this.filepath);
        ArffSaver saver = new ArffSaver();  // 用来保存的类
        saver.setInstances(data);
        // 用StringBuffer修改字符串，以便于保存文件
        StringBuffer tmp = new StringBuffer(this.filepath);
        int len = this.filepath.length();  // 得到长度
        tmp.delete(tmp.indexOf(".") + 1, len).append("arff");  // 将原文件类型删除并添加arff
        this.newFilepath = tmp.toString();

        saver.setFile(new File(newFilepath));
        saver.writeBatch();
        System.out.println("arff文件转换成功");
        ConfirmData(this.newFilepath);  // 实例化
        System.out.println("实例化成功");
        //System.out.println(this.data);  // 输出当前数据
    }

    public void DataPreProcess() throws Exception {  // 数据预处理
        // ** 从数据集中可以得到某些列只有同一个数据值，日期的具体意义不清楚，在这里选择去除
        String[] options = new String[2];
        options[0] = "-R";  // 删除指令
        options[1] = "1, 2, 7, 10, 12";  // 删除的列序号
        Remove re = new Remove();
        re.setOptions(options);
        re.setInputFormat(this.data);
        this.data = Filter.useFilter(this.data, re);  // 使用过滤器删除特征
        this.data.setClassIndex(data.numAttributes() - 1);
        int num = this.data.numAttributes();
        for (int i = 0; i < num; ++i) {
            System.out.println(data.attribute(i));
        }

        // ** 将数据规范到0-1区间
        Normalize norm = new Normalize();
        norm.setInputFormat(this.data);
        data = Filter.useFilter(this.data, norm);
    }

    public void AttributeSelect() throws Exception {
        // 需要选择属性的评估方法和搜素算法
        AttributeSelection select = new AttributeSelection();
        CfsSubsetEval eval = new CfsSubsetEval();
        BestFirst search = new BestFirst();  // 贪婪搜素，最好优先原则
        select.setEvaluator(eval);  // 设定评估算法
        select.setSearch(search);
        select.SelectAttributes(this.data);

        // 打印选择后的特征
        int[] attr = select.selectedAttributes();
        //System.out.println(this.data);
        System.out.println("选择后的特征");
        for (int idx: attr) {
            Attribute attribute = data.attribute(idx);
            System.out.println(attribute.name() + "  ");
        }
    }

    public void TrainTestSplit() {
        // 打乱数据集
        this.data.randomize(new Random(0));
        int rate = (int) Math.round(data.numInstances() * 0.60);  // 将数据集中的实例数 * 0.6
        // 构造两个Instance实例
        this.store[0] = new Instances(data, 0, rate);
        this.store[1] = new Instances(data, rate, data.numInstances() - rate);
    }

    // 使用决策树
    public void useJ48() throws Exception {
        this.startTime = System.currentTimeMillis();
        J48 tree = new J48();  // 经过测试J48无法处理numeric类型数据，因此在这里对数据进行类型转换

        store[0] = ChangeDataClass(store[0]);
        store[1] = ChangeDataClass(store[1]);

        tree.buildClassifier(store[0]);
        Evaluation eval = new Evaluation(this.store[0]);
        eval.evaluateModel(tree, store[1]);
        this.endTime = System.currentTimeMillis();
        System.out.println("======决策树J48结果======");
        System.out.println("运行时间" + (endTime - startTime) + "ms");
        System.out.println(eval.toClassDetailsString());  // 输出查全率、查准率等
        System.out.println(eval.toMatrixString() + "\n");  // 输出混淆矩阵
    }

    // 使用随机森林
    public void useRandomForest() throws Exception {
        // 使用随机森林存在内存不足的问题，先测试如果将数据进行类型转换是否可行
        this.startTime = System.currentTimeMillis();
        // ** 经过类型转换用符号代替内存还是会爆
        // ** 所以随机森林使用了数据离散化并进行了类型转换，经测试可行
        store[0] = DiscretizeData(store[0]);
        store[1] = DiscretizeData(store[1]);
        store[0] = ChangeDataClass(store[0]);
        store[1] = ChangeDataClass(store[1]);

        RandomForest rf = new RandomForest();
        rf.buildClassifier(store[0]);
        Evaluation eval = new Evaluation(store[0]);
        eval.evaluateModel(rf, store[1]);
        this.endTime = System.currentTimeMillis();
        System.out.println("======随机森林结果======");
        System.out.println("运行时间" + (endTime - startTime) + "ms");  // 输出时间
        System.out.println(eval.toClassDetailsString());
        System.out.println(eval.toMatrixString() + "\n");
    }

    // 使用朴素贝叶斯
    public void useNaiveBayes() throws Exception {
        this.startTime = System.currentTimeMillis();
        NaiveBayes nb = new NaiveBayes();
        nb.buildClassifier(store[0]);
        Evaluation eval = new Evaluation(store[0]);
        eval.evaluateModel(nb, store[1]);
        this.endTime = System.currentTimeMillis();
        System.out.println("======朴素贝叶斯结果======");
        System.out.println("运行时间" + (endTime - startTime) + "ms");
        System.out.println(eval.toClassDetailsString());
        System.out.println(eval.toMatrixString() + "\n");
    }

    // 使用神经网络
    public void useMultilayerPerceptron() throws Exception {
//        store[0] = ChangeDataClass(store[0]);
//        store[1] = ChangeDataClass(store[1]);

        this.startTime = System.currentTimeMillis();
        MultilayerPerceptron mp = new MultilayerPerceptron();
        mp.buildClassifier(store[0]);
        Evaluation eval = new Evaluation(store[0]);
        eval.evaluateModel(mp, store[1]);
        this.endTime = System.currentTimeMillis();
        System.out.println("======神经网络结果======");
        System.out.println("运行时间" + (endTime - startTime) + "ms");
        System.out.println(eval.toClassDetailsString());
        System.out.println(eval.toMatrixString() + "\n");
    }
}
