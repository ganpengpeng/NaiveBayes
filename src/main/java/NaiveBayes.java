import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.log4j.Logger;

import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

public class NaiveBayes {
    private static final Logger logger = Logger.getLogger(NaiveBayes.class);
    private Configuration conf;
    private Map<Path, Integer> classes;
    private Map<Path, Double> prior;
    private int docsTotalNum;

    public NaiveBayes(Configuration conf) {
        this.conf = conf;
        this.classes = new HashMap<>();
        this.prior = new HashMap<>();
    }

    public static void main(String[] args) throws Exception {
        Configuration conf = new Configuration();
        conf.set("fs.defaultFS", "hdfs://localhost:8020");
        NaiveBayes nb = new NaiveBayes(conf);
        nb.getTrainFile(new Path(args[0]));
        nb.calcPrior();
        for (Path path : nb.classes.keySet()) {
            Job job = Job.getInstance(conf, path.getName());
            job.setJarByClass(NaiveBayes.class);
            job.setMapperClass(MyMapper.class);
            job.setCombinerClass(MyReducer.class);
            job.setReducerClass(MyReducer.class);
            job.setOutputKeyClass(Text.class);
            job.setOutputValueClass(IntWritable.class);
            FileInputFormat.setInputPaths(job, path);
            FileOutputFormat.setOutputPath(job, new Path(args[1] + "/" + path.getName()));
            job.submit();
        }
    }

    public void getTrainFile(Path path) throws IOException {
        FileSystem fs = FileSystem.get(conf);
        FileStatus[] dirs = fs.listStatus(path);
        for (FileStatus dir : dirs) {
            FileStatus[] docs = fs.listStatus(dir.getPath());
            this.docsTotalNum += docs.length;
            classes.put(dir.getPath(), docs.length);
        }
    }

    public void calcPrior() {
        for (Map.Entry<Path, Integer> entry : classes.entrySet()) {
            double prior = entry.getValue() / (double) this.docsTotalNum;
            this.prior.put(entry.getKey(), prior);
            logger.info("class: " + entry.getKey().getName() + " prior: " + prior);
        }
    }
}

