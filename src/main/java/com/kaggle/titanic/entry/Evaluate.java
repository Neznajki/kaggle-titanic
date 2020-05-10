package com.kaggle.titanic.entry;

import com.kaggle.titanic.concurency.Processor;
import com.kaggle.titanic.data.machine.learning.Storage;
import com.kaggle.titanic.handler.EvaluationHandler;
import com.kaggle.titanic.helper.Resources;
import org.apache.log4j.BasicConfigurator;

import java.io.File;
import java.util.concurrent.ThreadLocalRandom;

public class Evaluate {
    public static void main(String[] args) throws Exception {
        BasicConfigurator.configure();

        Storage storage = new Storage(
            new Resources("test_file.csv"),
            1,
            2,
            500
        );

        storage.initForTest(0, ',');

        Processor processor = new Processor(storage);

        processor.initProcessEval(
            1,
            0.01,
            0.001,
            ThreadLocalRandom.current().nextInt(10, 1000)
        );

        EvaluationHandler evaluationHandler = new EvaluationHandler(processor, storage.getAllData());

        evaluationHandler.handleEval();

        evaluationHandler.listResults(storage.getRecordReader());
    }
}
