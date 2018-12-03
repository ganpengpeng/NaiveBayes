import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.*;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.log4j.Logger;

import java.io.*;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;

public class NaiveBayes {
    private static final Logger logger = Logger.getLogger(NaiveBayes.class);
    Configuration conf;
    // map<class, List<document>>: document for train
    Map<String, ArrayList<String>> classes;
    // document for test
    Map<String, ArrayList<String>> testFile;
    // prior probability
    Map<String, Double> prior;
    // total document number for train
    int docsTotalNum;
    // the number of all words in a class
    Map<String, Integer> totalWordsNumByClass;
    // Map<class, Map<word, number>>: the number of a word in a specific class
    Map<String, Map<String, Integer>> wordsNumByClass;

    public NaiveBayes(Configuration conf) {
        /*
         *  constructor for a new classifier.
         */
        this.conf = conf;
        this.classes = new HashMap<>();
        this.testFile = new HashMap<>();
        this.prior = new HashMap<>();
        this.docsTotalNum = 0;
        this.totalWordsNumByClass = new HashMap<>();
        this.wordsNumByClass = new HashMap<>();
    }

    public NaiveBayes(Configuration conf, String resultFile) throws Exception {
        /*
         *  constructor for a exist classifier in a file.
         *  this constructor for load a classifier which
         *  was serialized to a file before.
         *  conf: config for a file system.
         *  resultDir: path of the classifier.
         */
        FileSystem fs = FileSystem.get(conf);
        FSDataInputStream in = fs.open(new Path(resultFile));
        ObjectInputStream is = new ObjectInputStream(in);
        // read classifier form trainResult in resultDir.
        this.conf = conf;
        this.classes = (Map<String, ArrayList<String>>) is.readObject();
        this.testFile = (Map<String, ArrayList<String>>) is.readObject();
        this.prior = (Map<String, Double>) is.readObject();
        this.docsTotalNum = (int) is.readObject();
        this.totalWordsNumByClass = (Map<String, Integer>) is.readObject();
        this.wordsNumByClass = (Map<String, Map<String, Integer>>) is.readObject();
        is.close();
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
        NaiveBayes nb = new NaiveBayes(conf);
        nb.splitDataSet(new Path(args[0]));
        nb.calcPrior();
        nb.startMRJob(args);
        nb.trainAllClasses(new Path(args[1]));
        nb.save(args);
    }

    private void splitDataSet(Path path) throws IOException {
        /*
         *  split data set into train data and test data.
         *  randomly choose for 1/10 test and 9/10 train data.
         */
        FileSystem fs = FileSystem.get(this.conf);
        FileStatus[] dirs = fs.listStatus(path);
        for (FileStatus dir : dirs) {
            FileStatus[] docs = fs.listStatus(dir.getPath());
            ArrayList<String> trainPath = new ArrayList<>();
            ArrayList<String> testPath = new ArrayList<>();
            for (FileStatus doc : docs) {
                if (Math.random() > 0.1) {
                    trainPath.add(doc.getPath().getName());
                } else {
                    testPath.add(doc.getPath().getName());
                }
            }
            this.docsTotalNum += trainPath.size();
            this.classes.put(dir.getPath().getName(), trainPath);
            this.testFile.put(dir.getPath().getName(), testPath);
        }
    }

    private void calcPrior() {
        /*
         *  calculate prior of classes.
         *  I use laplace smoothing to deal with zero probability.
         */
        for (Map.Entry<String, ArrayList<String>> entry :
            this.classes.entrySet()) {
            //laplace smoothing
            double prior = (entry.getValue().size() + 1) /
                (double) (this.docsTotalNum + this.classes.size());
            this.prior.put(entry.getKey(), Math.log(prior));
            logger.info("class: " + entry.getKey() + " prior: " + prior);
        }
    }

    private void startMRJob(String[] args) throws Exception {
        FileSystem fs = FileSystem.get(this.conf);
        fs.delete(new Path(args[1]), true);
        for (String className : this.classes.keySet()) {
            Job job = Job.getInstance(this.conf, className);
            job.setJarByClass(NaiveBayes.class);
            job.setMapperClass(MyMapper.class);
            job.setCombinerClass(MyReducer.class);
            job.setReducerClass(MyReducer.class);
            job.setOutputKeyClass(Text.class);
            job.setOutputValueClass(IntWritable.class);
            FileInputFormat.setInputPaths(job,
                new Path(args[0] + "/" + className));
            FileOutputFormat.setOutputPath(job,
                new Path(args[1] + "/" + className));
            logger.info(job.waitForCompletion(true));
        }
    }

    private void trainAllClasses(Path output) throws Exception {
        FileSystem fs = FileSystem.get(this.conf);
        FileStatus[] fileStatuses = fs.listStatus(output);
        for (FileStatus status : fileStatuses) {
            if (status.isDirectory()) {
                train(status.getPath());
            }
        }
    }

    private void train(Path jobResultByClass) throws Exception {
        Map<String, Integer> wordsNumMap = new HashMap<>();
        int totalNum = 0;
        FileSystem fs = FileSystem.get(this.conf);
        FileStatus[] fileStatus = fs.listStatus(jobResultByClass,
            path -> path.getName().startsWith("part-r"));
        for (FileStatus status : fileStatus) {
            FSDataInputStream in = fs.open(status.getPath());
            InputStreamReader isr = new InputStreamReader(in);
            BufferedReader reader = new BufferedReader(isr);
            String oneLine = reader.readLine();
            while (oneLine != null) {
                String[] line = oneLine.split("\\s");
                try {
                    Integer num = Integer.valueOf(line[1]);
                    totalNum += num;
                    wordsNumMap.put(line[0], num);
                } catch (ArrayIndexOutOfBoundsException e) {
                    System.out.println("---ArrayIndexOutOfBoundsException---");
                    System.out.println(line.length);
                    System.out.println(line[0]);
                }
                oneLine = reader.readLine();
            }
        }
        this.totalWordsNumByClass.put(jobResultByClass.getName(), totalNum);
        this.wordsNumByClass.put(jobResultByClass.getName(), wordsNumMap);
    }

    String classify(Path doc) throws Exception {
        FileSystem fs = FileSystem.get(this.conf);
        FSDataInputStream is = fs.open(doc);
        InputStreamReader isr = new InputStreamReader(is);
        BufferedReader reader = new BufferedReader(isr);
        Map<String, Double> similarity = new HashMap<>(this.prior);
        String word;
        while ((word = reader.readLine()) != null) {
            for (Map.Entry<String, Map<String, Integer>> entry :
                this.wordsNumByClass.entrySet()) {
                Integer wordCount = entry.getValue().get(word);
                if (wordCount == null)
                    wordCount = 0;
                double sim = (wordCount + 1) / (double)
                    (this.totalWordsNumByClass.get(entry.getKey()) +
                        entry.getValue().size());
                similarity.computeIfPresent(entry.getKey(),
                    (key, oldValue) -> oldValue + Math.log(sim));
            }
        }
        ArrayList<Map.Entry<String, Double>> list =
            new ArrayList<>(similarity.entrySet());
        String result = list.get(0).getKey();
        Double sim = list.get(0).getValue();
        for (int i = 1; i < list.size(); i++) {
            Double aSim = list.get(i).getValue();
            if (aSim > sim) {
                sim = aSim;
                result = list.get(i).getKey();
            }
        }
        return result;
    }

    private void save(String[] args) throws Exception {
        FileSystem fs = FileSystem.get(this.conf);
        FSDataOutputStream out = fs.create(
            new Path(args[1] + "/trainResult"));
        ObjectOutputStream os = new ObjectOutputStream(out);
        os.writeObject(this.classes);
        os.writeObject(this.testFile);
        os.writeObject(this.prior);
        os.writeObject(this.docsTotalNum);
        os.writeObject(this.totalWordsNumByClass);
        os.writeObject(this.wordsNumByClass);
        os.close();
    }
}