package com.kaggle.titanic.data.machine.learning;

import com.kaggle.titanic.helper.Resources;
import org.datavec.api.records.metadata.RecordMetaData;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;

import java.io.File;
import java.io.IOException;
import java.util.concurrent.ThreadLocalRandom;

public class Storage {

    private final Resources resource;
    private final int labelIndex;
    private final int numClasses;
    private final int batchSize;
    private int numLinesToSkip = 0;
    private char delimiter = ',';
    private int seed;
    private Double trainDataPart = 0.8;
    RecordReader recordReader;
    DataSet allData;
    SplitTestAndTrain splitTestAndTrain;
    private RecordReaderDataSetIterator iterator;
    private DataNormalization normalizer;

    public Storage(
        Resources resource,
        int labelIndex,     //28 columns // results in 29
        int numClasses,     //2 possible answers 0 or 1
        int batchSize
    ) {
        this.resource = resource;
        this.labelIndex = labelIndex;
        this.numClasses = numClasses;
        this.batchSize = batchSize;
    }

    public void initForTest(
        int numLinesToSkip,
        char delimiter
    ) throws IOException, InterruptedException {
        this.numLinesToSkip = numLinesToSkip;
        this.delimiter = delimiter;

        recordReader = new CSVRecordReader(numLinesToSkip, delimiter);
        recordReader.initialize(new FileSplit(
            this.resource.getFile()
        ));

        this.readDataSet();
    }

    public void init(
        int numLinesToSkip,
        char delimiter
    ) throws IOException, InterruptedException {
        this.numLinesToSkip = numLinesToSkip;
        this.delimiter = delimiter;

        recordReader = new CSVRecordReader(numLinesToSkip, delimiter);
        recordReader.initialize(new FileSplit(
            this.resource.getFile()
        ));

        this.readDataSet();

//        allData.shuffle(this.seed);

        splitTestAndTrain = allData.splitTestAndTrain(1);
        splitTestAndTrain.getTrain();
    }

    public void setSeed(int seed) {
        this.seed = seed;
    }

    public void setTrainDataPart(Double trainDataPart) {
        this.trainDataPart = trainDataPart;
    }

    public void outputTrainData() {
        System.out.println("  +++++ Training Set Examples MetaData +++++");
        String format = "%-20s\t%s";

        for (RecordMetaData recordMetaData : splitTestAndTrain.getTrain().getExampleMetaData(RecordMetaData.class)) {
            System.out.println(String.format(format, recordMetaData.getLocation(), recordMetaData.getURI()));
        }
    }

    public void outputTestData() {
        System.out.println("  +++++ Test Set Examples MetaData +++++");

        for (RecordMetaData recordMetaData : splitTestAndTrain.getTest().getExampleMetaData(RecordMetaData.class)) {
            System.out.println(recordMetaData.getLocation());
        }
    }

    public void normalizeData() {
        normalizer = new NormalizerStandardize();
        normalizer.fit(splitTestAndTrain.getTrain());           //Collect the statistics (mean/stdev) from the training data. This does not modify the input data
        normalizer.transform(splitTestAndTrain.getTrain());     //Apply normalization to the training data
        normalizer.transform(splitTestAndTrain.getTest());         //Apply normalization to the test data. This is using statistics calculated from the *training* set
    }

    public DataSet getAllData() {
        return allData;
    }

    public DataSet getTrainData() {
        return allData;
    }

    public DataSet getTestData() {
        return this.splitTestAndTrain.getTest();
    }

    public RecordReaderDataSetIterator getIterator() {
        return iterator;
    }

    public DataNormalization getNormalizer() {
        return normalizer;
    }

    public RecordReader getRecordReader() {
        return recordReader;
    }

    private void readDataSet() {
        iterator = new RecordReaderDataSetIterator(recordReader, batchSize, labelIndex, numClasses);
        iterator.setCollectMetaData(true);
        allData = iterator.next();
    }
}
