import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;

public class NaiveBayesTest {
    private NaiveBayes nb;
    private Map<String, Integer[][]> classMatrix;
    private int testDocsNum;

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
        NaiveBayesTest nbt = new NaiveBayesTest(conf, args[1] + "/trainResult");
        nbt.test(args[0]);
        for (double v : nbt.macroAverage()) {
            System.out.println(v);
        }
        System.out.println();
        for (double v : nbt.microAverage()) {
            System.out.println(v);
        }
        System.out.println();
    }

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
//                System.out.println(docPath.toString() + ": " + resultClass);
            }
            this.testDocsNum += entry.getValue().size();
        }
        fillMatrix();
    }

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

    private double[] macroAverage() {
        double precision = 0, recall = 0;
        for (Integer[][] matrix : classMatrix.values()) {
            System.out.println(matrix[0][0] + ", " + matrix[0][1]
                + ", " + matrix[1][0] + ", " + matrix[1][1]);
            precision += matrix[0][0] / (double) (matrix[0][0] + matrix[0][1]);
            recall += matrix[0][0] / (double) (matrix[0][0] + matrix[1][0]);
        }
        double f1 = 2 * precision * recall / (precision + recall);
        return new double[]{precision, recall, f1};
    }

    private double[] microAverage() {
        Integer[][] matrix = {{0, 0}, {0, 0}};
        for (Integer[][] value : this.classMatrix.values()) {
            System.out.println(value[0][0] + ", " + value[0][1]
                + ", " + value[1][0] + ", " + value[1][1]);
            matrix[0][0] += value[0][0];
            matrix[0][1] += value[0][1];
            matrix[1][0] += value[1][0];
            matrix[1][1] += value[1][1];
        }
        double precision = matrix[0][0] / (double) (matrix[0][0] + matrix[0][1]);
        double recall = matrix[0][0] / (double) (matrix[0][0] + matrix[1][0]);
        double f1 = 2 * precision * recall / (precision + recall);
        return new double[]{precision, recall, f1};
    }
}
