package bayes;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.PairFunction;
import org.apache.spark.mllib.classification.NaiveBayes;
import org.apache.spark.mllib.classification.NaiveBayesModel;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.regression.LabeledPoint;
import scala.Tuple2;

/**
 * Created with IntelliJ IDEA.
 * User: sssd
 * Date: 2018/4/2 13:48
 * Version: V1.0
 * To change this template use File | Settings | File Templates.
 * Description:  java + spark mllib + bayes 测试文本分类
 */
public class TestBayes {

    public static void main(String[] args) {
        SparkConf sparkConf = new SparkConf();
        SparkConf conf = sparkConf.setAppName("ByesTest").setMaster("local[*]");
        JavaSparkContext jsc = new JavaSparkContext(conf);
        String trainPath = "file:///C:/Users/sssd/Desktop/bayes/trainData.txt";
        String testPath = "file:///C:/Users/sssd/Desktop/bayes/testData.txt";
        JavaRDD<String> lines = jsc.textFile(trainPath);
        JavaRDD<LabeledPoint> data = lines.map(new Function<String, LabeledPoint>() {
            public LabeledPoint call(String s) throws Exception {
                String[] split = s.split(",");
                String[] trainx = split[1].split(" ");

                LabeledPoint point = new LabeledPoint(Double.parseDouble(split[0]), Vectors.dense(Double.parseDouble(trainx[0]), Double.parseDouble(trainx[1]), Double.parseDouble(trainx[2])));
                return point;
            }
        });

        JavaRDD<LabeledPoint>[] split = data.randomSplit(new double[]{0.7, 0.3}, 11l);
        JavaRDD<LabeledPoint> trainData = split[0];
        JavaRDD<LabeledPoint> testData = split[1];
        final NaiveBayesModel model = NaiveBayes.train(trainData.rdd(), 1.0, "multinomial");
        JavaPairRDD<Double, Double> predictAndLabel = testData.mapToPair(new PairFunction<LabeledPoint, Double, Double>() {
            public Tuple2<Double, Double> call(LabeledPoint labeledPoint) throws Exception {
                return new Tuple2<Double, Double>(model.predict(labeledPoint.features()), labeledPoint.label());
            }
        });

        double error = predictAndLabel.filter(new Function<Tuple2<Double, Double>, Boolean>() {
            public Boolean call(Tuple2<Double, Double> doubleDoubleTuple2) throws Exception {

                return doubleDoubleTuple2._1.equals(doubleDoubleTuple2._2);
            }
        }).count() / (double) testData.count();

        System.out.println("预测的精度为：" + error);

/*        //方案二：保存模型后，导入模型，再计算，计算代码与方案一完全相同。
        model.save(jsc.sc(), "C://hello//BayesModel");
        NaiveBayesModel sameModel = NaiveBayesModel.load(jsc.sc(), "C://hello//BayesModel");
        //代码与方案一相同  */
    }

}
