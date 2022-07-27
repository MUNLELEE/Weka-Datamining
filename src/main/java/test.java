import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.CSVLoader;
import weka.core.converters.ConverterUtils;

import java.io.File;

public class test {
    public static void main(String[] args) throws Exception {
        // 分类
//        Dataclassify dc = new Dataclassify("D:\\沿途\\入学实验\\实验1\\实验数据\\电费回收数据.csv");
//        dc.Csv2Arff();
//        dc.DataPreProcess();
//        dc.AttributeSelect();
//        dc.TrainTestSplit();
//        dc.useJ48();
//        dc.useNaiveBayes();
//        dc.useRandomForest();
//        dc.useMultilayerPerceptron();

        // 回归
        Dataregression dr = new Dataregression("D:\\沿途\\入学实验\\实验1\\实验数据\\配网抢修数据.csv");
        dr.Csv2Arff();
        dr.DataPreProcess();
        dr.AttributeSelect();
//        dr.TrainTestSplit();
//        dr.useLinearRegression();
//        dr.useRandomForest();
//        dr.useMultilayerPercetron();
        dr.useApriori(0.3);
        dr.useApriori(0.6);
        dr.useApriori(0.9);

        // 聚类
//        Datacluster dc = new Datacluster("D:\\沿途\\入学实验\\实验1\\实验数据\\移动客户数据表.csv");
//        dc.Csv2Arff();
//        dc.DataPreProcess();
//        dc.AttributeSelect();
//        dc.GenerateClusterData();
//        dc.useEM();
//        dc.useKMeans();
//        dc.useHierarchicalCluster();
    }
}
