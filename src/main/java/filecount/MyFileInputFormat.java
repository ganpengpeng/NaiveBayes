package filecount;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.InputSplit;
import org.apache.hadoop.mapreduce.RecordReader;
import org.apache.hadoop.mapreduce.TaskAttemptContext;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.FileSplit;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;

public class MyFileInputFormat extends FileInputFormat<Text, IntWritable> {

    public RecordReader<Text, IntWritable> createRecordReader(InputSplit split,
                                                              TaskAttemptContext context)
        throws IOException, InterruptedException {
        return new MyReader();
    }

    public static class MyReader extends RecordReader<Text, IntWritable> {
        private BufferedReader in;
        private Text key = new Text();
        private IntWritable value = new IntWritable();

        public void initialize(InputSplit inputSplit, TaskAttemptContext context)
            throws IOException, InterruptedException {
            FileSplit split = (FileSplit) inputSplit;
            Configuration conf = context.getConfiguration();
            Path path = split.getPath();
            FileSystem fs = path.getFileSystem(conf);
            FSDataInputStream fsInput = fs.open(path);
            this.in = new BufferedReader(new InputStreamReader(fsInput));
        }

        public boolean nextKeyValue() throws IOException, InterruptedException {
            String s = in.readLine();
            if (s == null)
                return false;
            this.key.set(s);
            this.value.set(1);
            return true;
        }

        public Text getCurrentKey() throws IOException, InterruptedException {
            return this.key;
        }

        public IntWritable getCurrentValue() throws IOException, InterruptedException {
            return this.value;
        }

        public float getProgress() throws IOException, InterruptedException {
            return 0.5f;
        }

        public void close() throws IOException {
            in.close();
        }
    }
}
