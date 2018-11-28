import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;


public class NaiveBayes {
    private Configuration conf;
    private Map<Path, ArrayList<Path>> classes;
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
        nb.getTrainFile(new Path("Country"));


//        FileSystem fs;
//        fs = FileSystem.get(conf);
//        FSDataInputStream in = null;
//        try {
//            in = fs.open(new Path(args[0]));
//            IOUtils.copyBytes(in, System.out, 4096, false);
//        } finally {
//            IOUtils.closeStream(in);
//        }

//        Job job = Job.getInstance(conf, "naivebayes");
//        job.setJarByClass(NaiveBayes.class);
//
//        job.setMapperClass(MyMapper.class);
//        job.setCombinerClass(MyReducer.class);
//        job.setReducerClass(MyReducer.class);
//
//        job.setOutputKeyClass(Text.class);
//        job.setOutputValueClass(IntWritable.class);
//
//        FileInputFormat.setInputPaths(job, new Path(args[0]));
//        FileOutputFormat.setOutputPath(job, new Path(args[1]));
//        System.exit(job.waitForCompletion(true) ? 0 : 1);

    }

    public void getTrainFile(Path path) throws IOException {
        FileSystem fs = FileSystem.get(conf);
        FileStatus[] dirs = fs.listStatus(path);
        for (FileStatus dir : dirs) {
            FileStatus[] docs = fs.listStatus(dir.getPath());
            this.docsTotalNum += docs.length;
            ArrayList<Path> docPaths = new ArrayList<>();
            for (FileStatus doc : docs) {
                docPaths.add(doc.getPath());
            }
            classes.put(dir.getPath(), docPaths);
        }
    }

    public void calcPrior() {
        for (Map.Entry<Path, ArrayList<Path>> entry : classes.entrySet()) {
            double prior = entry.getValue().size() / (double) this.docsTotalNum;
            this.prior.put(entry.getKey(), prior);
            
        }
    }
}

