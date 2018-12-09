import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;

public class NaiveBayesTest {
    private NaiveBayes nb;
    private Map<String, Integer[][]> classMatrix;
    private int testDocsNum;

    /**
     * constructor fot NaiveBayesTest,
     * the classifier will be loaded inside constructor
     * @param conf conf for job
     * @param resultFile the file stored the classifier
     * @throws Exception
     */
    public NaiveBayesTest(Configuration conf, String resultFile) throws Exception {
        this.nb = new NaiveBayes(conf, resultFile);
        this.classMatrix = new HashMap<>();
        for (String s : this.nb.totalWordsNumByClass.keySet()) {
            Integer[][] matrix = {{0, 0}, {0, 0}};
            this.classMatrix.put(s, matrix);
        }
    }

    public static void main(String[] args) throws Exception {
        if (args.length != 2) {
            System.out.println("Usage: NaiveBayes \"data dir\" \"output dir\"");
            return;
        }
        Configuration conf = new Configuration();
        if (System.getProperty("user.name").equals("peng")) {
            conf.set("fs.defaultFS", "hdfs://localhost:8020");
        }
        NaiveBayesTest nbt = new NaiveBayesTest(conf, args[1] + "/Classifier");
        nbt.test(args[0]);
        // macroAverage result
        for (double v : nbt.macroAverage()) {
            System.out.println(v);
        }
        System.out.println();
        // microAverage result
        for (double v : nbt.microAverage()) {
            System.out.println(v);
        }
        System.out.println();
    }

    /**
     * test each file in NaiveBayes.testFile
     * @param dataset dataset
     * @throws Exception
     */
    private void test(String dataset) throws Exception {
        for (Map.Entry<String, ArrayList<String>> entry : this.nb.testFile.entrySet()) {
            String trueClass = entry.getKey();
            for (String testDoc : entry.getValue()) {
                Path docPath = new Path(dataset + "/" + trueClass + "/" + testDoc);
                String resultClass = this.nb.classify(docPath);
                Integer[][] matrix = this.classMatrix.get(resultClass);
                if (trueClass.equals(resultClass)) {
                    matrix[0][0] += 1;
                } else {
                    matrix[0][1] += 1;
                }
            }
            this.testDocsNum += entry.getValue().size();
        }
        fillMatrix();
    }

    /**
     * fill class matrix according to information from train result
     */
    private void fillMatrix() {
        for (Map.Entry<String, Integer[][]> entry : classMatrix.entrySet()) {
            ArrayList<String> testFile = this.nb.testFile.get(entry.getKey());
            int docsNum;
            if (testFile != null) {
                docsNum = testFile.size();
            } else {
                docsNum = 0;
            }
            Integer[][] matrix = entry.getValue();
            matrix[1][0] = docsNum - matrix[0][0];
            matrix[1][1] = this.testDocsNum - docsNum - matrix[0][1];
        }
    }

    /**
     * calculate evaluation by macro average
     * @return [precision, recall, f1]
     */
    private double[] macroAverage() {
        double precision = 0, recall = 0;
        for (Integer[][] matrix : this.classMatrix.values()) {
            System.out.println(matrix[0][0] + ", " + matrix[0][1]
                + ", " + matrix[1][0] + ", " + matrix[1][1]);
            precision += matrix[0][0] / (double) (matrix[0][0] + matrix[0][1]);
            recall += matrix[0][0] / (double) (matrix[0][0] + matrix[1][0]);
        }
        precision /= this.classMatrix.size();
        recall /= this.classMatrix.size();
        double f1 = 2 * precision * recall / (precision + recall);
        return new double[]{precision, recall, f1};
    }

    /**
     * calculate evaluation by micro average
     * @return [precision, recall, f1]
     */
    private double[] microAverage() {
        Integer[][] matrix = {{0, 0}, {0, 0}};
        for (Integer[][] value : this.classMatrix.values()) {
            matrix[0][0] += value[0][0];
            matrix[0][1] += value[0][1];
            matrix[1][0] += value[1][0];
            matrix[1][1] += value[1][1];
        }
        System.out.println("----"+matrix[0][0] + ", " + matrix[0][1]
            + ", " + matrix[1][0] + ", " + matrix[1][1]);
        double precision = matrix[0][0] / (double) (matrix[0][0] + matrix[0][1]);
        double recall = matrix[0][0] / (double) (matrix[0][0] + matrix[1][0]);
        double f1 = 2 * precision * recall / (precision + recall);
        return new double[]{precision, recall, f1};
    }
}
