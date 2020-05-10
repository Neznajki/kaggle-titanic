package com.kaggle.titanic.handler;

import com.kaggle.titanic.concurency.Processor;
import org.datavec.api.records.Record;
import org.datavec.api.records.metadata.RecordMetaData;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.writable.Writable;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.ListDataSetIterator;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.evaluation.meta.Prediction;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class EvaluationHandler {

    private final Processor processor;
    private final DataSet testData;
    private Evaluation eval;
    INDArray output;

    public EvaluationHandler(Processor processor, DataSet testData) {

        this.processor = processor;
        this.testData = testData;
    }

    public void handleEval() {
        eval = processor.getModel().evaluate(new ListDataSetIterator<>(testData.asList()));
        output = processor.getModel().output(testData.getFeatures());
        eval.eval(
            testData.getLabels(),
            output,
            testData.getExampleMetaData(RecordMetaData.class)
        );          //Note we are passing in the test set metadata here
        System.out.println(eval.stats(false, true));
    }

    public void printErrors(
        RecordReaderDataSetIterator iterator,
        DataNormalization normalizer,
        RecordReader recordReader
    ) throws IOException {
        //Get a list of prediction errors, from the Evaluation object
        //Prediction errors like this are only available after calling iterator.setCollectMetaData(true)
        List<Prediction> predictionErrors = eval.getPredictionErrors();
        System.out.println("\n\n+++++ Prediction Errors +++++");
        for (Prediction p : predictionErrors) {
            System.out.println("Predicted class: " + p.getPredictedClass() + ", Actual class: " + p.getActualClass()
                + "\t" + p.getRecordMetaData(RecordMetaData.class).getLocation());
        }

        //We can also load a subset of the data, to a DataSet object:
        List<RecordMetaData> predictionErrorMetaData = new ArrayList<>();
        for (Prediction p : predictionErrors) predictionErrorMetaData.add(p.getRecordMetaData(RecordMetaData.class));
        DataSet predictionErrorExamples = iterator.loadFromMetaData(predictionErrorMetaData);
        normalizer.transform(predictionErrorExamples);  //Apply normalization to this subset

        //We can also load the raw data:
        List<Record> predictionErrorRawData = recordReader.loadFromMetaData(predictionErrorMetaData);

        //Print out the prediction errors, along with the raw data, normalized data, labels and network predictions:
        for (int i = 0; i < predictionErrors.size(); i++) {
            Prediction p = predictionErrors.get(i);
            RecordMetaData meta = p.getRecordMetaData(RecordMetaData.class);
            INDArray features = predictionErrorExamples.getFeatures().getRow(i, true);
            INDArray labels = predictionErrorExamples.getLabels().getRow(i, true);
            List<Writable> rawData = predictionErrorRawData.get(i).getRecord();

            INDArray networkPrediction = processor.getModel().output(features);

            System.out.println(meta.getLocation() + ": "
                + "\tRaw Data: " + rawData
                + "\tNormalized: " + features
                + "\tLabels: " + labels
                + "\tPredictions: " + networkPrediction);
        }
    }

    public void listResults(RecordReader recordReader) throws IOException {
        printPassengers(recordReader, 0);
        printPassengers(recordReader, 1);
    }

    private void printPassengers(RecordReader recordReader, int survived) throws IOException {
        List<RecordMetaData> predictionMetaData = new ArrayList<>();
        for (Prediction prediction : eval.getPredictionByPredictedClass(survived)) {
            predictionMetaData.add(prediction.getRecordMetaData(RecordMetaData.class));
        }
        List<Record> predictionRawData = recordReader.loadFromMetaData(predictionMetaData);

        for (Record record : predictionRawData) {
            System.out.println(record.getRecord().get(0) + ", " + survived);
        }
    }
}
