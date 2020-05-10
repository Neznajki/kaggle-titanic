package com.kaggle.titanic.entry;

import com.kaggle.titanic.concurency.Processor;
import com.kaggle.titanic.data.machine.learning.Storage;
import com.kaggle.titanic.handler.EvaluationHandler;
import com.kaggle.titanic.helper.Resources;
import org.apache.log4j.BasicConfigurator;
import org.datavec.api.records.Record;
import org.datavec.api.records.metadata.RecordMetaData;
import org.datavec.api.writable.Writable;
import org.deeplearning4j.datasets.iterator.impl.ListDataSetIterator;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.evaluation.meta.Prediction;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;

import java.io.File;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ThreadLocalRandom;

public class MachineLearning {
    public static void main(String[] args) throws Exception {
        BasicConfigurator.configure();

        Storage storage = new Storage(
            new Resources("train_file.csv"),
            1,
            2,
            900
        );

        storage.setSeed(ThreadLocalRandom.current().nextInt(10, 1000));
        storage.init(0, ',');
        Processor processor = new Processor(storage);

        processor.initProcess(
            10000,
            0.0005,
            0.0002,
            ThreadLocalRandom.current().nextInt(10, 1000)
        );
        processor.learn(5000);

        EvaluationHandler evaluationHandler = new EvaluationHandler(processor, storage.getTestData());

        evaluationHandler.handleEval();

//        evaluationHandler.listResults(storage.getRecordReader());
//        List<Prediction> list2 = eval.getPredictionByPredictedClass(2);     //All predictions for predicted class 2
//        List<Prediction> list3 = eval.getPredictionsByActualClass(2);       //All predictions for actual class 2
    }
}
