package filecount;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.InputSplit;
import org.apache.hadoop.mapreduce.RecordReader;
import org.apache.hadoop.mapreduce.TaskAttemptContext;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.FileSplit;

import java.io.IOException;

public class MyFileInputFormat extends FileInputFormat<Text, IntWritable> {

    public RecordReader<Text, IntWritable> createRecordReader(InputSplit split,
                                                              TaskAttemptContext context)
        throws IOException, InterruptedException {
        return new MyReader();
    }

    public static class MyReader extends RecordReader<Text, IntWritable> {
        private Text key = new Text();
        private IntWritable value = new IntWritable();
        private Path filePath;
        private boolean flag;

        public void initialize(InputSplit inputSplit, TaskAttemptContext context)
            throws IOException, InterruptedException {
            FileSplit split = (FileSplit) inputSplit;
            Configuration conf = context.getConfiguration();
            this.filePath = split.getPath();
            System.err.println(filePath);
        }

        public boolean nextKeyValue() throws IOException, InterruptedException {
            if (this.flag) {
                this.key = null;
                this.value = null;
                return false;
            }
            String className = filePath.getParent().getName();
            this.key.set(className);
            this.value.set(1);
            this.flag = true;
            return true;
        }

        public Text getCurrentKey() throws IOException, InterruptedException {
            return this.key;
        }

        public IntWritable getCurrentValue() throws IOException, InterruptedException {
            return this.value;
        }

        public float getProgress() throws IOException, InterruptedException {
            return 0;
        }

        public void close() throws IOException {

        }
    }
}
