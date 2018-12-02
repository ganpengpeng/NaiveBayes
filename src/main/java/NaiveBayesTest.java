import org.apache.hadoop.conf.Configuration;

public class NaiveBayesTest {
    private NaiveBayes nb;

    public NaiveBayesTest(Configuration conf, String resultFile) throws Exception {
        nb = new NaiveBayes(conf, resultFile);
    }

    public static void main(String[] args) throws Exception {
        if (args.length != 2) {
            System.out.println("Usage: NaiveBayes \"data dir\" \"output dir\"");
            return;
        }
        Configuration conf = new Configuration();
        if (System.getProperty("user.name").equals("peng")){
            conf.set("fs.defaultFS", "hdfs://localhost:8020");
        }
        NaiveBayesTest nbt = new NaiveBayesTest(conf, args[1] + "/trainResult");
        System.out.println(nbt.nb.docsTotalNum);
        System.out.println(nbt.nb.wordsNumByClass.get("USA").entrySet());
    }

    private String test(String s) {
        return s;
    }
}
