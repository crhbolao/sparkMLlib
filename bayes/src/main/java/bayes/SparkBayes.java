package bayes;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.PairFunction;
import org.apache.spark.broadcast.Broadcast;
import org.apache.spark.mllib.classification.NaiveBayes;
import org.apache.spark.mllib.classification.NaiveBayesModel;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.storage.StorageLevel;
import org.deeplearning4j.models.embeddings.loader.VectorsConfiguration;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.deeplearning4j.spark.text.functions.TextPipeline;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import scala.Tuple2;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Created with IntelliJ IDEA.
 * User: sssd
 * Date: 2018/4/2 14:29
 * Version: V1.0
 * To change this template use File | Settings | File Templates.
 * Description:  java + sparkMLlib + bayes 实现中文文本的分类
 */
public class SparkBayes implements Serializable{

    private static Integer maxlength = 0;

    private static Broadcast<Map<String, Object>> broadcasTokenizerVarMap;

    private static AtomicInteger lineNum = new AtomicInteger(0);

    public void entryPoint(String[] args) throws Exception {

        SparkConf sparkConf = new SparkConf();
        SparkConf conf = sparkConf.setAppName("ByesTest").setMaster("local[*]");
        JavaSparkContext jsc = new JavaSparkContext(conf);

        String inputPath = "file:///C:/Users/sssd/Desktop/LSTM/data.txt";
        String savePath = "file:///C:/Users/sssd/Desktop/LSTM/bayesModel.txt";

        JavaRDD<LabeledPoint> dataSet = getDataSet(jsc, inputPath);
        JavaRDD<LabeledPoint>[] split = dataSet.randomSplit(new double[]{0.1, 0.9}, 11l);
        JavaRDD<LabeledPoint> trainData = split[0];
        JavaRDD<LabeledPoint> testData = split[1];

        final NaiveBayesModel model = NaiveBayes.train(trainData.rdd());
        JavaPairRDD<Double, Double> predictAndLabel = testData.mapToPair(new PairFunction<LabeledPoint, Double, Double>() {
            public Tuple2<Double, Double> call(LabeledPoint labeledPoint) throws Exception {
                System.out.println("预测的结果为：" + model.predict(labeledPoint.features()) + " 真实的结果为：" + labeledPoint.label());
                return new Tuple2<Double, Double>(model.predict(labeledPoint.features()), labeledPoint.label());
            }
        });

        double error = predictAndLabel.filter(new Function<Tuple2<Double, Double>, Boolean>() {
            public Boolean call(Tuple2<Double, Double> doubleDoubleTuple2) throws Exception {

                return doubleDoubleTuple2._1.equals(doubleDoubleTuple2._2);
            }
        }).count() / (double) testData.count();

        System.out.println("预测的精度为：" + error);

    }

    public static JavaRDD<LabeledPoint> getDataSet(JavaSparkContext jsc, String inputPath) {
        JavaRDD<Tuple2<String, String>> sourceData = readFile(jsc, inputPath).persist(StorageLevel.MEMORY_AND_DISK_SER_2());
        JavaRDD<String> label = readLabel(sourceData);
        JavaRDD<String> text = readText(sourceData);
        initTokenizer(jsc);
        JavaRDD<List<VocabWord>> labelList = pipeLine(label);
        JavaRDD<List<VocabWord>> textList = pipeLine(text);
        findMaxlength(textList);
        JavaRDD<Tuple2<List<VocabWord>, VocabWord>> combine = combine(labelList, textList);
        JavaRDD<LabeledPoint> data = chang2LabeledPoint(combine).persist(StorageLevel.MEMORY_AND_DISK_SER_2());
        sourceData.unpersist();
        return data;
    }

    public static JavaRDD<Tuple2<String, String>> readFile(JavaSparkContext jsc, String path) {
        JavaRDD<String> javaRDD = jsc.textFile(path);
        JavaRDD<Tuple2<String, String>> rdd = javaRDD.map(new Function<String, Tuple2<String, String>>() {
            public Tuple2<String, String> call(String s) throws Exception {
                String[] split = s.split("\t");
                Tuple2<String, String> tuple2 = new Tuple2<String, String>(split[0], split[1]);
                return tuple2;
            }
        });
        return rdd;
    }

    public static JavaRDD<String> readLabel(JavaRDD<Tuple2<String, String>> data) {
        JavaRDD<String> labelRdd = data.map(new Function<Tuple2<String, String>, String>() {
            public String call(Tuple2<String, String> stringStringTuple2) throws Exception {
                String label = stringStringTuple2._1;
                return label;
            }
        });
        return labelRdd;
    }

    public static JavaRDD<String> readText(JavaRDD<Tuple2<String, String>> data) {
        JavaRDD<String> textRdd = data.map(new Function<Tuple2<String, String>, String>() {
            public String call(Tuple2<String, String> stringStringTuple2) throws Exception {
                String text = stringStringTuple2._2;
                return text;
            }
        });
        return textRdd;
    }

    public static void initTokenizer(JavaSparkContext jsc) {
        Map<String, Object> TokenizerVarMap = new HashMap<String, Object>();
        TokenizerVarMap.put("numWords", 1);     //min count of word appearence
        TokenizerVarMap.put("nGrams", 1);       //language model parameter
        TokenizerVarMap.put("tokenizer", DefaultTokenizerFactory.class.getName());  //tokenizer implemention
        TokenizerVarMap.put("tokenPreprocessor", CommonPreprocessor.class.getName());
        TokenizerVarMap.put("useUnk", true);    //unlisted words will use usrUnk
        TokenizerVarMap.put("vectorsConfiguration", new VectorsConfiguration());
        TokenizerVarMap.put("stopWords", new ArrayList<String>());  //stop words
        broadcasTokenizerVarMap = jsc.broadcast(TokenizerVarMap);
    }

    public static JavaRDD<List<VocabWord>> pipeLine(JavaRDD<String> javaRDDText) {
        JavaRDD<List<VocabWord>> textVocab = null;
        try {
            TextPipeline pipeline = new TextPipeline(javaRDDText, broadcasTokenizerVarMap);
            pipeline.buildVocabCache();
            pipeline.buildVocabWordListRDD();
            textVocab = pipeline.getVocabWordListRDD();
        } catch (Exception e) {
            e.printStackTrace();
        }
        return textVocab;
    }

    public static void findMaxlength(JavaRDD<List<VocabWord>> textList) {
        JavaRDD<Integer> map = textList.map(new Function<List<VocabWord>, Integer>() {
            public Integer call(List<VocabWord> vocabWords) throws Exception {
                int size = vocabWords.size();
                if (maxlength < size) {
                    maxlength = size;
                }
                return size;
            }
        });
        map.count();
        System.out.println("预料的最大维度为：" + maxlength);
    }

    public static JavaRDD<Tuple2<List<VocabWord>, VocabWord>> combine(JavaRDD<List<VocabWord>> label, JavaRDD<List<VocabWord>> text) {
        final List<List<VocabWord>> labels = label.collect();
        JavaRDD<Tuple2<List<VocabWord>, VocabWord>> res = text.map(new Function<List<VocabWord>, Tuple2<List<VocabWord>, VocabWord>>() {
            public Tuple2<List<VocabWord>, VocabWord> call(List<VocabWord> vocabWords) throws Exception {
                int num = lineNum.getAndIncrement();
                VocabWord labelVocabWord = labels.get(num).get(0);
                Tuple2<List<VocabWord>, VocabWord> tuple2 = new Tuple2<List<VocabWord>, VocabWord>(vocabWords, labelVocabWord);
                if (lineNum.equals(labels.size() - 1)) {
                    lineNum = new AtomicInteger(0);
                }
                return tuple2;
            }
        });
        return res;
    }

    public static JavaRDD<LabeledPoint> chang2LabeledPoint(JavaRDD<Tuple2<List<VocabWord>, VocabWord>> combine) {
        JavaRDD<LabeledPoint> res = combine.map(new Function<Tuple2<List<VocabWord>, VocabWord>, LabeledPoint>() {
            public LabeledPoint call(Tuple2<List<VocabWord>, VocabWord> tuple) throws Exception {
                List<VocabWord> wordList = tuple._1;
                VocabWord labelList = tuple._2;
                int datay = labelList.getIndex();

                double[] datax = new double[maxlength];
                for (int i = 0; i < maxlength; i++) {
                    if (i < wordList.size()) {
                        VocabWord temp = wordList.get(i);
                        datax[i] = (double) temp.getIndex();
                    } else {
                        datax[i] = (0.0);
                    }
                }
                return new LabeledPoint((double) datay, Vectors.dense(datax));
            }
        });
        return res;
    }

    public static void main(String[] args) throws Exception {
        new SparkBayes().entryPoint(args);
    }

}
