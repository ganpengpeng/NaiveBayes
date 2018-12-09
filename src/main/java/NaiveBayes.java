import filecount.MyFileInputFormat;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.*;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.log4j.Logger;
import wordcount.WordCount;
import wordcount.WordCountMapper;
import wordcount.WordCountReducer;

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

    /**
     * default constructor
     *
     * @param conf the conf from main function
     */
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

    /**
     * @param conf       config for a file system.
     * @param resultFile the file which store the classifier
     * @throws Exception exception from ObjectInputStream.readObject
     */
    public NaiveBayes(Configuration conf, String resultFile) throws Exception {
        /*
         *  constructor from a exist classifier in a file.
         *  this constructor is used to load a classifier which
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

    /**
     * @param args args[0]: dataset directory
     *             args[1]: output directory for job result
     * @throws Exception
     */
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
        // auto split data to train data and test data
        nb.splitDataSet(new Path(args[0]));
        // submit document count job and calculate prior probability
        nb.calcPrior(args);
        // start word count job for each class
        nb.startMRJob(args);
        // train classifier
        nb.trainAllClasses(new Path(args[1]));
        // serialize the classifier to hdfs
        nb.save(args);
    }

    /**
     * @param path dataset path
     * @throws IOException exception from FileSystem.listStatus
     */
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

    /**
     * @param args args in main function
     * @throws IOException            just throw
     * @throws InterruptedException   just throw
     * @throws ClassNotFoundException just throw
     */
    private void calcPrior(String[] args) throws IOException,
        InterruptedException, ClassNotFoundException {
        /*
         *  calculate prior of classes.
         *  I use laplace smoothing to deal with zero probability.
         */
        Configuration conf = this.conf;
        /*
         *  this argument is set to let fileinputformat to iterate recursively
         *  to each document.
         */
        conf.set("mapreduce.input.fileinputformat.input.dir.recursive", "true");
        Path output = new Path("filecount");
        FileSystem fs = FileSystem.get(conf);
        fs.delete(output, true);
        // document count job
        Job job = Job.getInstance(this.conf, "filecount");
        job.setJarByClass(NaiveBayes.class);
        job.setMapperClass(WordCount.TokenizerMapper.class);
        job.setCombinerClass(WordCount.IntSumReducer.class);
        job.setReducerClass(WordCount.IntSumReducer.class);
        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(IntWritable.class);
        job.setInputFormatClass(MyFileInputFormat.class);
        FileInputFormat.setInputPaths(job, new Path(args[0]));
        FileOutputFormat.setOutputPath(job, output);
        /*
         *******************************************************
         *  how to distinguish test file from train file?
         *  As we know, the above job count every file in class directory,
         *  but I do not divide test file from train file, so the return
         *  value will be total number of documents of class. But what we
         *  need is just the number of train files of each class.
         *  I use a trick here to get the actual number of train files.
         *  That is using total number to minus the number of test files
         *  in this.testFile(NaiveBayes.testFile).
         *  (Ps: using map reduce job will not always speed up our program!)
         *******************************************************
         */
        Map<String, Integer> docNum = new HashMap<>();
        int docTotalNum = 0;
        if (job.waitForCompletion(true)) {
            FSDataInputStream in = fs.open(new Path("filecount/part-r-00000"));
            BufferedReader reader = new BufferedReader(new InputStreamReader(in));
            String s = reader.readLine();
            while (s != null) {
                String[] line = s.split("\\s");
                Integer num = Integer.valueOf(line[1]);
                // trick below
                num -= this.testFile.get(line[0]).size();
                docNum.put(line[0], num);
                docTotalNum += num;
                s = reader.readLine();
            }
        } else {
            logger.info("waitForCompletion false!!!");
            return;
        }
        // calculate prior probability for each class
        for (Map.Entry<String, Integer> entry : docNum.entrySet()) {
            double prior = (entry.getValue() + 1) /
                (double) (docTotalNum + docNum.size());
            this.prior.put(entry.getKey(), Math.log(prior));
        }
    }

    /**
     * start word count for each class
     *
     * @param args args in main fucntion
     * @throws Exception just throw
     */
    private void startMRJob(String[] args) throws Exception {
        FileSystem fs = FileSystem.get(this.conf);
        fs.delete(new Path(args[1]), true);
        // one class, one word count job
        for (String className : this.classes.keySet()) {
            Job job = Job.getInstance(this.conf, className);
            job.setJarByClass(NaiveBayes.class);
            job.setMapperClass(WordCountMapper.class);
            job.setCombinerClass(WordCountReducer.class);
            job.setReducerClass(WordCountReducer.class);
            job.setOutputKeyClass(Text.class);
            job.setOutputValueClass(IntWritable.class);
            FileInputFormat.setInputPaths(job,
                new Path(args[0] + "/" + className));
            FileOutputFormat.setOutputPath(job,
                new Path(args[1] + "/" + className));
            logger.info(job.waitForCompletion(true));
        }
    }

    /**
     * train class according to word count job result
     *
     * @param output word count output path
     * @throws Exception
     */
    private void trainAllClasses(Path output) throws Exception {
        FileSystem fs = FileSystem.get(this.conf);
        FileStatus[] fileStatuses = fs.listStatus(output);
        for (FileStatus status : fileStatuses) {
            if (status.isDirectory()) {
                train(status.getPath());
            }
        }
    }

    /**
     * train the specific class
     *
     * @param jobResultByClass job output path by specific class
     * @throws Exception
     */
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

    /**
     * this is the function to classify a document
     *
     * @param doc the document to classify
     * @return result class
     * @throws Exception
     */
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
                // probability of a class
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
        // get the maximum probability
        for (int i = 1; i < list.size(); i++) {
            Double aSim = list.get(i).getValue();
            if (aSim > sim) {
                sim = aSim;
                result = list.get(i).getKey();
            }
        }
        return result;
    }

    /**
     * serialize the classifier to hdfs file.
     *
     * @param args args in main function
     * @throws Exception
     */
    private void save(String[] args) throws Exception {
        FileSystem fs = FileSystem.get(this.conf);
        FSDataOutputStream out = fs.create(
            new Path(args[1] + "/Classifier"));
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