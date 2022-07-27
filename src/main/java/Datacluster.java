import weka.associations.Apriori;
import weka.attributeSelection.AttributeSelection;
import weka.attributeSelection.BestFirst;
import weka.attributeSelection.CfsSubsetEval;
import weka.classifiers.Evaluation;
import weka.clusterers.ClusterEvaluation;
import weka.clusterers.EM;
import weka.clusterers.HierarchicalClusterer;
import weka.clusterers.SimpleKMeans;
import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.ConverterUtils;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Normalize;
import weka.filters.unsupervised.attribute.Remove;

import java.io.File;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Random;

public class Datacluster {
    private Instances data;
    private String filepath;
    private String newFilepath;
    private long startTime, endTime;

    Datacluster(String filepath) {
        this.filepath = filepath;
    }

    private void ConfirmData(String filepath) throws Exception {
        this.data = new Instances(new FileReader(filepath));
    }

    // 计算类内距离函数
    private double[] CalcDistanceIn(ArrayList<double[]> data) {
        int sample_num = data.size();
        double[] distance = new double[sample_num];
        int attrs = data.get(0).length;
        for (int i = 0; i < sample_num; ++i) {  // 对于每个被选取的样本
            double dis = 0;
            for (int j = 0; j < sample_num && j != i; ++j) {  // 对于类内的其他的样本
                double dis_tmp = 0;
                for (int k = 0; k < attrs; ++k) {  // 对于每个属性
                    dis_tmp += Math.pow((data.get(i)[k] - data.get(j)[k]), 2);  // 计算距离
                }
                dis_tmp = Math.sqrt(dis_tmp);  // 和一个样本的距离
                dis += dis_tmp;  // 选取的样本和类内所有样本的总距离
            }
            distance[i] = dis;
        }
        return distance;
    }

    // 计算类外距离函数
    private ArrayList<double[]> CalcDistanceOut(ArrayList<double[]> data1, ArrayList<double[]> data2) {
        int sam_num1 = data1.size();
        int sam_num2 = data2.size();
        int attr = data1.get(0).length;
        double[] dis_out_1 = new double[sam_num1];  // 存储每个样本的类外总距离
        double[] dis_out_2 = new double[sam_num2];
        Arrays.fill(dis_out_1, 0.0);
        Arrays.fill(dis_out_2, 0.0);
        for (int i = 0; i < sam_num1; ++i) {
            double dis = 0;  // 选取的样本的总距离
            for (int j = 0; j < sam_num2; ++j) {
                // 循环非同类样本
                double dis_tmp = 0;
                for (int k = 0; k < attr; ++k) {
                    // 循环每个特征
                    dis_tmp += Math.pow((data1.get(i)[k] - data2.get(j)[k]), 2);
                }
                dis_tmp = Math.sqrt(dis_tmp);
                dis += dis_tmp;
                dis_out_2[j] += dis_tmp;  // 距离是双向的，所以提前存储第二类的距离
            }
            dis_out_1[i] = dis;
        }
        ArrayList<double[]> res = new ArrayList<>(2);
        res.add(dis_out_1);
        res.add(dis_out_2);
        return res;
    }

    // 计算轮廓系数的函数
    private double CalcSilhouette(double[] label) throws Exception {
        int len = label.length;  // 数据总数
        double res = 0;  // 最后的轮廓系数结果
        ArrayList<double[]> pive = new ArrayList<>();  // 存储正样本
        ArrayList<double[]> neve = new ArrayList<>();  // 存储负样本
        for (int i = 0; i < len; ++i) {
            if (label[i] == 0) {
                neve.add(this.data.get(i).toDoubleArray());
            }
            else {
                pive.add(this.data.get(i).toDoubleArray());
            }
        }
        int pive_sum = pive.size(), neve_sum = neve.size();  // 两类簇的数量
        int attrs = pive.get(0).length;  // 特征数

        double[] dist_in_pive = new double[pive_sum];  // 正类，类内的样本距离
        double[] dist_in_neve = new double[neve_sum];
        dist_in_neve = CalcDistanceIn(neve);
        dist_in_pive = CalcDistanceIn(pive);

        // 计算a(t)
        for (int i = 0; i < pive_sum; ++i) {
            dist_in_pive[i] /= pive_sum - 1;
        }
        for (int i = 0; i < neve_sum; ++i) {
            dist_in_neve[i] /= neve_sum - 1;
        }

        ArrayList<double[]> distancsOut = new ArrayList<>(2);
        distancsOut = CalcDistanceOut(pive, neve);  // 先1后0
        // 求平均
        for (int i = 0; i < pive_sum; ++i) {
            distancsOut.get(0)[i] /= neve_sum;
        }
        for (int i = 0; i < neve_sum; ++i) {
            distancsOut.get(1)[i] /= neve_sum;
        }
        double[] silNum = new double[len];  // 存储每个样本的轮廓系数
        int idx = 0;  // 用来最后存储的索引
        for (int i = 0; i < pive_sum; ++i, ++idx) {
            silNum[idx] = (distancsOut.get(0)[i] - dist_in_pive[i]) / Math.max(distancsOut.get(0)[i], dist_in_pive[i]);
        }
        for (int i = 0; i < neve_sum; ++i, ++idx) {
            silNum[idx] = (distancsOut.get(1)[i] - dist_in_neve[i]) / Math.max(distancsOut.get(1)[i], dist_in_neve[i]);
        }
        for (int i = 0; i < len; ++i) {
            res += silNum[i];
        }
        res /= len;
        return res;
    }

    // 重载，计算单个类的方差
    private ArrayList<double[]> CalcSquare(ArrayList<double[]> data1, ArrayList<double[]> data0, int attrs) {

        ArrayList<double[]> res = new ArrayList<>();  // 返回两类方差向量

        double[] res_0 = new double[attrs];  // 存储两类的方差向量
        double[] res_1 = new double[attrs];
        int sum_data1 = data1.size();  // 1类的总数
        int sum_data0 = data0.size();  // 0类的总数
        // 计算0类，先计算均值后计算方差
        for (int i = 0; i < attrs; ++i) {  // 循环每个特征
            double tmp_res = 0;
            for (int j = 0; j < sum_data0; ++j) {
                tmp_res += data0.get(j)[i];
            }
            tmp_res /= sum_data0;
            res_0[i] = tmp_res;
        }
        for (int i = 0; i < attrs; ++i) {
            double tmp_res = 0;
            for (int j = 0; j < sum_data0; ++j) {
                tmp_res += Math.pow((data0.get(j)[i] - res_0[i]), 2);
            }
            tmp_res /= sum_data0;
            res_0[i] = tmp_res;
        }

        // 计算1类
        for (int i = 0; i < attrs; ++i) {
            double tmp_res = 0;
            for (int j = 0; j < sum_data1; ++j) {
                tmp_res += data1.get(j)[i];
            }
            tmp_res /= sum_data1;
            res_1[i] = tmp_res;
        }
        for (int i = 0; i < attrs; ++i) {
            double tmp_res = 0;
            for (int j = 0; j < sum_data1; ++j) {
                tmp_res += Math.pow((data1.get(j)[i] - res_1[i]), 2);
            }
            tmp_res /= sum_data1;
            res_1[i] = tmp_res;
        }
        res.add(res_0);
        res.add(res_1);
        return res;  // 返回也是先0后1
    }

    // 计算总数据的方差
    private double[] CalcSquare(ArrayList<double[]> data, int attrs) {  // 传入总数据以及数据条数
        double[] res = new double[attrs];
        int sum_data = data.size();
        // 计算总数据的均值向量
        for (int i = 0; i < attrs; ++i) {  // 循环每个特征
            double tmp_sum = 0;
            for (int j = 0; j < sum_data; ++j) {  // 循环每条数据
                tmp_sum += data.get(j)[i];
            }
            tmp_sum /= sum_data;
            res[i] = tmp_sum;
        }
        for (int i = 0; i < attrs; ++i) {
            double tmp_sum = 0;  // 记录方差总和
            for (int j = 0; j < sum_data; ++j) {
                tmp_sum += Math.pow((data.get(j)[i] - res[i]), 2);
            }
            tmp_sum /= sum_data;
            res[i] = tmp_sum;
        }
        return res;  // 返回样本的总方差向量
    }

    // 计算第一部分的函数
    private double CalcScat(ArrayList<double[]> std_c, double[] std_all, int cNum) {  //类方差，总方差，类数
        double res = 0;
        int len = std_all.length;
        double dis_0 = 0, dis_1 = 0, dis = 0;  // 0类1类和总体二范数
        for (int i = 0; i < len; ++i) {
            dis += Math.pow(std_all[i], 2);
            dis_0 += Math.pow(std_c.get(0)[i], 2);
            dis_1 += Math.pow(std_c.get(1)[i], 2);
        }
        dis = Math.sqrt(dis);
        dis_0 = Math.sqrt(dis_0);
        dis_1 = Math.sqrt(dis_1);
        res = 0.5 * ((dis_0 / dis) + (dis_1 / dis));
        return res;
    }

    // 计算第二部分的函数
    // 参数：第一类数据，第二类数据，两类的中心向量，所有样本的中心向量
    private double CalcDensbw(ArrayList<double[]> data0, ArrayList<double[]> data1, ArrayList<double[]> cents,
                              double[] cent_all) {
        double res = 0;
        double mean_std = 0;  // 存储所有簇的平均标准差
        int attrs = cents.get(0).length;  // 特征数
        double dis_0 = 0, dis_1 = 0;
        for (int i = 0; i < attrs; ++i) {
            dis_0 += Math.pow(cents.get(0)[i], 2);
            dis_1 += Math.pow(cents.get(1)[i], 2);
        }
        mean_std = 0.5 * Math.sqrt(Math.sqrt(dis_0) + Math.sqrt(dis_1));  // ? 存疑
        int den_0 = 0, den_1 = 0;  // 存储两类的密度

        // 计算第0类
        int len0 = data0.size();
        for (int i = 0; i < len0; ++i) {  // 循环所有数据
            double dis_tmp = 0;
            for (int j = 0; j < attrs; ++j) {  // 循环所有特征
                dis_tmp += Math.pow((cents.get(0)[j] - data0.get(i)[j]), 2);
            }
            dis_tmp = Math.sqrt(dis_tmp);
            if (dis_tmp > mean_std) {
                ++den_0;
            }
        }
        // 计算第1类
        int len1 = data1.size();
        for (int i = 0; i < len1; ++i) {
            double dis_tmp = 0;
            for (int j = 0; j < attrs; ++j) {
                dis_tmp += Math.pow((cents.get(1)[j] - data1.get(i)[j]), 2);
            }
            dis_tmp = Math.sqrt(dis_tmp);
            if (dis_tmp > mean_std) {
                ++den_1;
            }
        }

        // 计算总密度
        int den_all = 0;
        for (int i = 0; i < len0; ++i) {
            double dis_tmp = 0;
            for (int j = 0; j < attrs; ++j) {
                dis_tmp += Math.pow((cent_all[j] - data0.get(i)[j]), 2);
            }
            dis_tmp = Math.sqrt(dis_tmp);
            if (dis_tmp > mean_std) {
                ++den_all;
            }
        }
        for (int i = 0; i < len1; ++i) {
            double dis_tmp = 0;
            for (int j = 0; j < attrs; ++j) {
                dis_tmp += Math.pow((cent_all[j] - data1.get(i)[j]), 2);
            }
            dis_tmp = Math.sqrt(dis_tmp);
            if (dis_tmp > mean_std) {
                ++den_all;
            }
        }
        res = (double)den_all / Math.max(den_0, den_1);
        return res;
    }

    // 计算S_Dbw，函数中调用计算方差的函数
    private double CalcSDbw(double[] label, ArrayList<double[]> cents) throws Exception {  // 参数：类别和各类中心向量
        double res = 0;
        int len = label.length;  // 数据条数
        ArrayList<double[]> pive = new ArrayList<>();  // 存储1类
        ArrayList<double[]> neve = new ArrayList<>();  // 存储0类
        for (int i = 0; i < len; ++i) {
            if (label[i] == 0) {
                neve.add(this.data.get(i).toDoubleArray());
            }
            else {
                pive.add(this.data.get(i).toDoubleArray());
            }
        }


        int attrs = pive.get(0).length;  // 特征数，作为之后函数的参数传入
        // 传入的cents是先0后1
        double[] cen_cluster = new double[attrs];  // 存储所有的中心，也就是将两个簇的中心平均
        for (int i = 0; i < attrs; ++i) {
            cen_cluster[i] = (cents.get(0)[i] + cents.get(1)[i]) / 2;
        }

        ArrayList<double[]> allData = new ArrayList<>();  // 将两类数据合并，作为调用方差的参数
        allData.addAll(pive);
        allData.addAll(neve);

        ArrayList<double[]> std_2c = new ArrayList<>(2);  // 存储两类的方差向量，先0后1
        std_2c = CalcSquare(pive, neve, attrs);
        double[] std_all = CalcSquare(allData, attrs);
        double res_1 = CalcScat(std_2c, std_all, 2);  // 存储第一部分的结果
        double res_2 = CalcDensbw(neve, pive, cents, cen_cluster);
        return res_2 + res_1;
    }
    // 用来生成无类数据
    public void GenerateClusterData() throws Exception {
        Remove re = new Remove();
        // attribute从1开始，index从0开始
//        System.out.println(this.data.numAttributes() + "***");
//        System.out.println(this.data.classIndex() + "###");
        re.setAttributeIndices("" + (this.data.classIndex() + 1));
        re.setInputFormat(this.data);
        this.data = Filter.useFilter(this.data, re);
    }

    public void Csv2Arff() throws Exception {
        // 源文件为tsv格式无法读取，在读取之前利用python将文件格式转换为csv格式
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
//        System.out.println(this.data);  // 源数据
    }

    public void DataPreProcess() throws Exception {
        String[] options = new String[2];
        // 删除一些认为没有用的信息
        options[0] = "-R";
        options[1] = "1, 2, 3, 4, 17, 72, 73";  // 具体删除了一些id类的信息
        Remove re = new Remove();
        re.setOptions(options);
        re.setInputFormat(this.data);
        this.data = Filter.useFilter(this.data, re);
        this.data.setClassIndex(this.data.numAttributes() - 1);

        // 数据归一化
        Normalize norm = new Normalize();
        norm.setInputFormat(this.data);
        this.data = Filter.useFilter(this.data, norm);

        this.data.randomize(new Random(0));
//        System.out.println(this.data);
    }

    public void AttributeSelect() throws Exception {
        AttributeSelection select = new AttributeSelection();
        CfsSubsetEval eval = new CfsSubsetEval();
        BestFirst search = new BestFirst();
        select.setEvaluator(eval);
        select.setSearch(search);
        select.SelectAttributes(this.data);

        int[] attr = select.selectedAttributes();
        System.out.println("选择后的特征");
        for (int idx: attr) {
            System.out.println(this.data.attribute(idx));
        }
    }

    public void useKMeans() throws Exception {
        this.startTime = System.currentTimeMillis();
        SimpleKMeans kMeans = new SimpleKMeans();
        kMeans.buildClusterer(this.data);
        ClusterEvaluation cEval = new ClusterEvaluation();
        cEval.setClusterer(kMeans);
        cEval.evaluateClusterer(this.data);

        Instances centers = kMeans.getClusterCentroids();
        int num = centers.numInstances();
        ArrayList<double[]> cents = new ArrayList<>();  // 存储每一类的中心向量
        for (int i = 0; i < num; ++i) {
            cents.add(centers.get(i).toDoubleArray());
        }
        double[] store = cEval.getClusterAssignments();
        double sDbw = CalcSDbw(store, cents);
        double sil = CalcSilhouette(store);
        this.endTime = System.currentTimeMillis();

        System.out.println("======KMeans聚类算法======");
        System.out.println("运行时间：" + (endTime - startTime) + "ms");
        System.out.println("轮廓系数为：" + sil);
        System.out.println("SDbw指标为：" + sDbw);
        System.out.println(cEval.clusterResultsToString());  // 得到聚类算法的大体情况
    }

    public void useEM() throws Exception {
        this.startTime = System.currentTimeMillis();
        EM em = new EM();
        em.setMaximumNumberOfClusters(2);  // 设置聚类的最大数目
        em.buildClusterer(this.data);
        ClusterEvaluation cEval = new ClusterEvaluation();
        cEval.setClusterer(em);
        cEval.evaluateClusterer(this.data);

        double[] store = cEval.getClusterAssignments();
        double sil = CalcSilhouette(store);

        // 没有相应的api可以知道中心向量，因此在这里手动计算
        int len = store.length;
        ArrayList<double[]> cents = new ArrayList<>(2);
        ArrayList<double[]> data0 = new ArrayList<>();  // 分别存放两类的数据
        ArrayList<double[]> data1 = new ArrayList<>();
        for (int i = 0; i < len; ++i) {
            if (store[i] == 0) {
                data0.add(this.data.instance(i).toDoubleArray());
            }
            else {
                data1.add(this.data.instance(i).toDoubleArray());
            }
        }
        int attrs = this.data.numAttributes();
        double[] attr_store = new double[attrs];  // 存储每个类的中心向量
        // 求第0类
        len = data0.size();
        for (int i = 0; i < attrs; ++i) {  // 循环每个特征
            double tmp = 0;
            for (int j = 0; j < len; ++j) {  // 循环每条数据
                tmp += data0.get(j)[i];
            }
            tmp /= len;
            attr_store[i] = tmp;
        }
        cents.add(attr_store);
        Arrays.fill(attr_store, 0.0);  // 清零
        // 求第2类
        len = data1.size();
        for (int i = 0; i < attrs; ++i) {
            double tmp = 0;
            for (int j = 0; j < len; ++j) {
                tmp += data1.get(j)[i];
            }
            tmp /= len;
            attr_store[i] = tmp;
        }
        cents.add(attr_store);

        double sDbw = CalcSDbw(store, cents);
        this.endTime = System.currentTimeMillis();

        System.out.println("运行时间：" + (endTime - startTime) + "ms");
        System.out.println("======EM算法======");
        System.out.println("轮廓系数为：" + sil);
        System.out.println("SDbw指标为：" + sDbw);
        System.out.println(cEval.clusterResultsToString());

    }

    public void useHierarchicalCluster() throws Exception {
        int rate = (int)Math.round(this.data.numInstances() * 0.05);
        this.data = new Instances(this.data, 0, rate);

        this.startTime = System.currentTimeMillis();
        HierarchicalClusterer hc = new HierarchicalClusterer();
//        hc.setNumClusters(2);
        hc.buildClusterer(this.data);
        System.out.println("类数：" + hc.getNumClusters());
//        System.out.println(hc.getRevision());
        ClusterEvaluation cEval = new ClusterEvaluation();
        cEval.setClusterer(hc);
        cEval.evaluateClusterer(this.data);

        double[] store = cEval.getClusterAssignments();
        double sil = CalcSilhouette(store);  // 计算轮廓系数

        // 得到两类簇的中心向量
        int len = store.length;  // 数据数量
        ArrayList<double[]> cents = new ArrayList<>(2);
        ArrayList<double[]> data0 = new ArrayList<>();  // 存放两类数据
        ArrayList<double[]> data1 = new ArrayList<>();
        for (int i = 0; i < len; ++i) {
            if (store[i] == 0) {
                data0.add(this.data.instance(i).toDoubleArray());
            }
            else {
                data1.add(this.data.instance(i).toDoubleArray());
            }
        }

        // 计算0类
        int attrs = this.data.numAttributes();
        len = data0.size();
        double[] attr_store = new double[attrs];  // 存储中心向量
        for (int i = 0; i < attrs; ++i) {
            double tmp = 0;
            for (int j = 0; j < len; ++j) {
                tmp += data0.get(j)[i];
            }
            tmp /= len;
            attr_store[i] = tmp;
        }
        cents.add(attr_store);

        // 求第1类
        Arrays.fill(attr_store, 0);
        len = data1.size();
        for (int i = 0; i < attrs; ++i) {
            double tmp = 0;
            for (int j = 0; j < len; ++j) {
                tmp += data1.get(j)[i];
            }
            tmp /= len;
            attr_store[i] = tmp;
        }
        cents.add(attr_store);
        double sDbw = CalcSDbw(store, cents);
        this.endTime = System.currentTimeMillis();

        System.out.println("运行时间" + (endTime - startTime) + "ms");
        System.out.println("======层次聚类结果======");
        System.out.println("轮廓系数为：" + sil);
        System.out.println("sDbw指标为：" + sDbw);
        System.out.println(cEval.clusterResultsToString());
    }

}
