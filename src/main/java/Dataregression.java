import weka.associations.Apriori;
import weka.attributeSelection.*;
import weka.classifiers.Evaluation;
import weka.classifiers.functions.LinearRegression;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.classifiers.trees.RandomForest;
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

public class Dataregression {
    // 类设置和数据分类基本一样
    private Instances data;
    private String filepath;
    private String newFilepath;
    private long startTime, endTime;
    private Instances[] store = new Instances[2];

    Dataregression (String filepath) {
        this.filepath = filepath;
    }

    private void ConfirmData(String filepath) throws Exception {
        this.data = new Instances(new FileReader(filepath));
    }

    public void Csv2Arff() throws Exception {
        Instances data = ConverterUtils.DataSource.read(this.filepath);
        ArffSaver saver = new ArffSaver();
        saver.setInstances(data);
        // 转换字符串
        StringBuffer tmp = new StringBuffer(this.filepath);
        int len = this.filepath.length();
        tmp.delete(tmp.indexOf(".") + 1, len).append("arff");
        this.newFilepath = tmp.toString();

        saver.setFile(new File(this.newFilepath));
        saver.writeBatch();
        System.out.println("arff文件转换成功");
        ConfirmData(this.newFilepath);
        System.out.println("实例化成功");
//        System.out.println(this.data);  // 打印源数据
    }

    public void DataPreProcess() throws Exception {
        // ** 从数据集中删除一些不需要的特征
        String[] options = new String[2];
        options[0] = "-R";
        options[1] = "1, 2";  // 删除日期和地区
        Remove re = new Remove();
        re.setOptions(options);
        re.setInputFormat(this.data);
        this.data = Filter.useFilter(this.data, re);

        int num = data.numAttributes();
        this.data.setClassIndex(num - 1);
        for (int i = 0; i < num; ++i) {
            System.out.println(this.data.attribute(i));
        }

        // 将数据规范到0-1区间
        Normalize norm = new Normalize();
        norm.setInputFormat(this.data);
        this.data = Filter.useFilter(this.data, norm);

        // 在分类测试中对随机森林进行了离散化处理，这里沿用
        // 寻找关联规则时没有使用
//        Discretize dis = new Discretize();
//        dis.setInputFormat(this.data);
//        this.data = Filter.useFilter(this.data, dis);
//        System.out.println(this.data);  // 预处理后
    }

    public void AttributeSelect() throws Exception {
        AttributeSelection select = new AttributeSelection();
        CfsSubsetEval eval = new CfsSubsetEval();
        GreedyStepwise search = new GreedyStepwise();
//        BestFirst search = new BestFirst();
        select.setEvaluator(eval);
        select.setSearch(search);
        select.SelectAttributes(this.data);

        int[] attr = select.selectedAttributes();
        System.out.println("选择后的特征");
        for (int idx: attr) {
            System.out.println(this.data.attribute(idx));
        }
    }

    public void TrainTestSplit() throws Exception {
        this.data.randomize(new Random(0));
        int rate = (int)Math.round(this.data.numInstances() * 0.60);
        this.store[0] = new Instances(this.data, 0, rate);
        this.store[1] = new Instances(this.data, rate, data.numInstances() - rate);
    }

    public void useLinearRegression() throws Exception {
        this.startTime = System.currentTimeMillis();
        LinearRegression linear = new LinearRegression();
        linear.buildClassifier(store[0]);
        Evaluation eval = new Evaluation(store[1]);
        eval.evaluateModel(linear, store[1]);
        this.endTime = System.currentTimeMillis();
        System.out.println("======线性回归结果======");
        System.out.println("运行时间 " + (endTime - startTime));
        System.out.println(eval.toSummaryString("model result", false));  // 输出RMSE、RAE
    }

    public void useRandomForest() throws Exception {
        this.startTime = System.currentTimeMillis();
        RandomForest rf = new RandomForest();
        rf.buildClassifier(store[0]);
        Evaluation eval = new Evaluation(store[1]);
        eval.evaluateModel(rf, store[1]);
        this.endTime = System.currentTimeMillis();
        System.out.println("======随机森林结果======");
        System.out.println("运行时间" + (endTime - startTime));
        System.out.println(eval.toSummaryString("model result", false));
    }

    public void useMultilayerPercetron() throws Exception {
        this.startTime = System.currentTimeMillis();
        MultilayerPerceptron mp = new MultilayerPerceptron();
        mp.buildClassifier(store[0]);
        Evaluation eval = new Evaluation(store[1]);
        eval.evaluateModel(mp, store[1]);
        this.endTime = System.currentTimeMillis();
        System.out.println("======神经网络结果======");
        System.out.println("运行时间" + (endTime - startTime));
        System.out.println(eval.toSummaryString("model result", false));
    }

    public void useApriori(double confidence) throws Exception {
        NumericToNominal ntn = new NumericToNominal();
        ntn.setInputFormat(this.data);
        this.data = Filter.useFilter(this.data, ntn);

        this.startTime = System.currentTimeMillis();
        Apriori apriori = new Apriori();
        String[] options = new String[2];
        options[0] = "-C";
        options[1] = Double.toString(confidence);
        apriori.setOptions(options);
        apriori.buildAssociations(this.data);
        this.endTime = System.currentTimeMillis();

        System.out.println("置信度为" + confidence + "时，运行时间为" + (this.endTime - this.startTime) + "ms");
        System.out.println(apriori);
    }
}
