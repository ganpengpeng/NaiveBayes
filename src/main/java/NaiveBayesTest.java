import org.apache.hadoop.conf.Configuration;

import java.util.HashMap;
import java.util.Map;

public class NaiveBayesTest {
    private NaiveBayes nb;
    private Map<String, Integer[][]> classMatrix;

    public NaiveBayesTest(Configuration conf, String resultFile) throws Exception {
        this.nb = new NaiveBayes(conf, resultFile);
        this.classMatrix = new HashMap<>();
        for (String s : this.nb.totalWordsNumByClass.keySet()) {
            this.classMatrix.put(s, new Integer[2][2]);
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
        System.out.println(nbt.nb.docsTotalNum);
        System.out.println(nbt.nb.wordsNumByClass.get("USA").entrySet());
    }

    private void test() {
        
        return;
    }
}
